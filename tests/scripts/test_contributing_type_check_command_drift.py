"""Mirror contract: CONTRIBUTING.md's Type-check bullet restates the live ``type_check_command``.

Task 4108. CONTRIBUTING.md documents a copy-pasteable pyright command inside a
``type-check-command-mirror`` HTML-comment marker. This guard pins the package
DIRECTORIES that command walks against the ones
``dark-factory-orchestrator.yaml``'s ``type_check_command`` actually checks.

WHY A MACHINE CHECK. The doc drifted while the config grew. Task 3397 widened
the yaml chain from 3 members to 7; the bullet's fenced command still read
``cd fused-memory && uv run pyright   # also: orchestrator, dashboard`` five
months later, naming 3. That drift happened while CONTRIBUTING.md ALREADY
carried its "Treat ``dark-factory-orchestrator.yaml``'s ``test_command`` /
``lint_command`` / ``type_check_command`` as the source of truth if these
drift" pointer — so it is MEASURED, not predicted, that prose pointing at prose
does not hold here. Task 4538 then corrected the surrounding SENTENCE by hand
(to "runs all seven workspace members") without touching the command, which is
further evidence for a mechanical tie rather than against one: the number and
the command went out of sync with each other.

THE HARM IS NOT COSMETIC. A contributor who runs the documented command
type-checks a strict SUBSET of what the merge gate checks, gets a clean LOCAL
GREEN, and then eats a RED merge verify on a branch they believed was ready.
Under-coverage is the specific failure this guard exists to stop; it is the same
harm class ``test_contributing_lint_command_drift.py`` (task 3558) was filed for
on the neighbouring Lint bullet, and this guard is deliberately built to that
one's shape.

WHAT IS PINNED, AND WHY IT IS NOT BYTE EQUALITY. The Lint mirror asserts verbatim
string equality because both its sides invoke ``uv run ruff check``. The two
type-check lanes deliberately DIVERGE at the runner: the merge gate runs ``npx
pyright`` (Node lane, repo-root ``package.json`` pin installed by ``npm ci``),
while a contributor runs ``uv run pyright`` (wheel lane, ``uv.lock``'s
pyright-python pin). CONTRIBUTING.md documents that divergence on purpose and
``test_pyright_version_pin.py`` (task 4538) holds both lanes to one version. So
this guard pins the SET AND ORDER OF PACKAGE DIRECTORIES — what actually drifted,
and what causes the local-green/gate-red split — and pins the runner SHAPE from
BOTH ends separately, so the deliberate divergence can never quietly collapse
into a byte copy of the gate command.

Order, not just set: order is the chain walk's semantics. A mis-tracked relative
``cd`` yields a wrong-but-same-set result that a set comparison would pass.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config
(``tests/scripts/orchestrator.yaml``), so this guard actually runs under
FULL_SUITE and merge-role ``merge_verify_breadth: full``. A guard that never ran
on merge verify would be vacuous the same way the drift it watches for is — the
rationale ``test_contributing_lint_command_drift.py`` and
``test_pyright_version_pin.py`` both state.

OUT OF SCOPE, and deliberately not pinned here. ``hooks/project-checks`` sets
``PYRIGHT_PACKAGES=(fused-memory orchestrator dashboard)`` — really three — so
CONTRIBUTING.md's §4 item 3 (``uv run pyright`` in each touched,
pyright-configured package) and its pre-commit "pyright up to 3x" sentence are
ACCURATE ABOUT THE HOOK and must stay that way. §4 item 3 is additionally the
DECOY this module's extractor is explicitly tested against below: pinning that
generic instruction to the live config would be wrong twice over — it would fail
immediately, and "fixing" it would destroy correct, audience-appropriate advice.

Like its Lint sibling, this guard asserts about two committed COMMAND artifacts
only. It makes no assertion about the surrounding sentences and must not be
extended to: it has to stay green if the prose around the marker is fully
reworded.
"""
from __future__ import annotations

import pathlib
import re
import shlex
import tomllib

import pytest
import verify_command_invariants as vci
import yaml

from orchestrator import verify_cmd

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The canonical root config filename. Loading it by this exact path means the
# guard fails loudly if the config is renamed or type_check_command disappears,
# rather than quietly guarding nothing.
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"

# The HTML-comment marker anchoring extraction, wrapped around the Type-check
# bullet's fenced command. An explicit marker rather than "the span matching
# `pyright`" because CONTRIBUTING.md carries SEVERAL other pyright spans — §4
# item 3's generic `uv run pyright` in each touched, pyright-configured package,
# and the pre-commit paragraph's "pyright up to 3x" — which are deliberately
# generic and must never be pinned. Any "first matching span" extractor would
# silently re-target on a doc reorder. The marker also puts the tie AT THE EDIT
# SITE, which is the root cause this task closes: nothing tied the prose to the
# config.
MIRROR_BEGIN = "type-check-command-mirror:begin"
MIRROR_END = "type-check-command-mirror:end"

# The fenced ```bash block inside the marked slice. Anchoring on the fence
# (rather than on "a backtick span") is what keeps the begin comment's own
# inline-code prose out of the extraction — the real marker cites
# `uv run pyright` and `type_check_command` in its explanatory text.
_MARKED_BASH_FENCE = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _documented_type_check_command(markdown_text: str) -> str:
    """The command in the fenced bash block delimited by the mirror markers.

    Every failure is a loud ``AssertionError`` naming the marker literal and
    CONTRIBUTING.md, never a ``''``/``None`` return. That is the vacuity hazard
    and the whole point: an extractor that silently yields nothing turns the
    drift assertion green while pinning nothing — strictly worse than no guard,
    because the check still reports success.

    Returns the command ``strip()``ed and otherwise verbatim. No further
    normalisation: the downstream comparison depends on not silently
    canonicalising away a real difference.
    """
    begin_count = markdown_text.count(MIRROR_BEGIN)
    assert begin_count == 1, (
        f"expected exactly one {MIRROR_BEGIN!r} marker, found {begin_count} "
        f"(task 4108). This marker delimits the Type-check bullet's fenced "
        f"command in CONTRIBUTING.md that mirrors the package directories of "
        f"dark-factory-orchestrator.yaml's type_check_command. If it was "
        f"deleted, restore it around that fenced block; if it was duplicated, "
        f"one of the two mirrors is unpinned and free to drift."
    )
    end_count = markdown_text.count(MIRROR_END)
    assert end_count == 1, (
        f"expected exactly one {MIRROR_END!r} marker to close {MIRROR_BEGIN!r} "
        f"in CONTRIBUTING.md, found {end_count} (task 4108) — restore the "
        f"closing marker below the Type-check bullet's fenced command"
    )

    # Inverted markers yield an empty slice, so the next assertion catches that
    # too, loudly and with the same remedy.
    marked = markdown_text[markdown_text.index(MIRROR_BEGIN) : markdown_text.index(MIRROR_END)]
    fences: list[str] = _MARKED_BASH_FENCE.findall(marked)
    assert len(fences) == 1, (
        f"expected exactly one fenced ```bash block between {MIRROR_BEGIN!r} and "
        f"{MIRROR_END!r} in CONTRIBUTING.md, found {len(fences)}: {fences!r} "
        f"(task 4108). The marker must wrap that fence and nothing else; if the "
        f"bullet was restructured or the markers were inverted, move the marker "
        f"back around the copy-pasteable type-check command."
    )

    lines = [line for line in fences[0].splitlines() if line.strip()]
    # ONE line, deliberately. `verify._cd_clause_target` recognises only an exact
    # two-token `cd <dir>`, so a multi-line or backslash-continued chain would
    # leave the walker unable to see the `cd` clauses — and this guard would then
    # pin half a chain while still reporting green. Fail loudly and say so.
    assert len(lines) == 1, (
        f"expected exactly ONE non-empty line inside the fenced ```bash block "
        f"between {MIRROR_BEGIN!r} and {MIRROR_END!r} in CONTRIBUTING.md, found "
        f"{len(lines)}: {lines!r} (task 4108). The documented chain must be a "
        f"single line with no backslash continuations: "
        f"orchestrator/src/orchestrator/verify.py::_cd_clause_target recognises "
        f"only an exact two-token `cd <dir>`, so a continued line would leave "
        f"the chain walker unable to see the `cd` clauses and this guard would "
        f"silently pin only part of the chain."
    )

    command = lines[0].strip()
    assert command, (
        f"the command between {MIRROR_BEGIN!r} and {MIRROR_END!r} in "
        f"CONTRIBUTING.md is empty (task 4108)"
    )
    return command


# Extractor fixtures are hand-written markdown, never the real CONTRIBUTING.md,
# so they stay stable under any future edit to that file's content. They spell
# the marker literals out in full rather than interpolating the constants above:
# a rename must not be able to silently keep a broken parser agreeing with its
# own fixtures.

# (a) Happy path, modelled on the real block: a fenced ```bash example ABOVE the
# marker, and — the measured hazard — inline-code spans inside the begin
# comment's own prose, which an extractor keyed on backticks alone would return.
_HAPPY_DOC = """\
- **Tests** run per-package with `pytest`, e.g.:
  ```bash
  cd orchestrator && uv run pytest tests/ --timeout=300
  ```
- **Type-check** (pyright) — the same members the merge gate checks:
  <!-- type-check-command-mirror:begin
       Mirrors the package directories walked by `type_check_command` in
       dark-factory-orchestrator.yaml. The RUNNER deliberately differs:
       `uv run pyright` here, `npx pyright` there. Pinned by
       tests/scripts/test_contributing_type_check_command_drift.py. -->
  ```bash
  cd alpha && uv run pyright && cd ../beta && uv run pyright
  ```
  <!-- type-check-command-mirror:end -->
  Either invocation works, and both resolve the same pinned version.
"""

_HAPPY_COMMAND = "cd alpha && uv run pyright && cd ../beta && uv run pyright"

# (b) No marker at all — deleted, or the bullet restructured. The doc still
# CONTAINS a plausible-looking fenced pyright command: the extractor must not
# fall back to "find something that looks right".
_NO_MARKER_DOC = """\
- **Type-check** (pyright):
  ```bash
  cd alpha && uv run pyright
  ```

3. `uv run pyright` in each touched, pyright-configured package.
"""

# (c) Two marker blocks — e.g. a section duplicated in a bad merge. Picking the
# first silently pins one mirror and lets the other rot unwatched.
_DUPLICATE_MARKER_DOC = """\
<!-- type-check-command-mirror:begin -->
```bash
cd alpha && uv run pyright
```
<!-- type-check-command-mirror:end -->

## Some later section

<!-- type-check-command-mirror:begin -->
```bash
cd delta && uv run pyright
```
<!-- type-check-command-mirror:end -->
"""

# (d) Inverted markers — :end above :begin. The slice between them is empty, so
# no fence is found and the fence assertion fires with the same remedy.
_INVERTED_MARKER_DOC = """\
<!-- type-check-command-mirror:end -->
```bash
cd alpha && uv run pyright
```
<!-- type-check-command-mirror:begin -->
"""

# (e) A marked fence carrying a continued, multi-line chain. Accepting it would
# hand the walker clauses in which `cd ../beta &&` is not an exact two-token
# `cd <dir>`, so the guard would pin a strict prefix of the chain while
# reporting green — the precise vacuity this extractor must refuse.
_MULTILINE_FENCE_DOC = """\
<!-- type-check-command-mirror:begin -->
```bash
cd alpha && uv run pyright && \\
  cd ../beta && uv run pyright
```
<!-- type-check-command-mirror:end -->
"""

# (f) Decoy immunity. CONTRIBUTING.md's REAL §4 item 3 line and its real
# pre-commit "pyright up to 3x" sentence appear BOTH before and after the marker
# block, so neither a "first span" nor a "last span" heuristic passes by
# accident. A MEASURED hazard, not hypothetical: CONTRIBUTING.md really carries
# both, and both are CORRECT about `hooks/project-checks`'s three-package
# `PYRIGHT_PACKAGES` — they must stay generic and unpinned.
_DECOY_DOC = """\
3. `uv run pyright` in each touched, pyright-configured package
   (`fused-memory`, `orchestrator`, `dashboard`).

```bash
cd fused-memory && uv run pyright   # also: orchestrator, dashboard
```

<!-- type-check-command-mirror:begin
     Mirrors the package directories walked by `type_check_command`. -->
```bash
cd alpha && uv run pyright && cd ../beta && uv run pyright
```
<!-- type-check-command-mirror:end -->

**`hooks/pre-commit`** additionally runs `ruff check` and **pyright up to 3x**
(once per touched package under `PYRIGHT_PACKAGES`).

3. `uv run pyright` in each touched, pyright-configured package.
"""


def test_documented_type_check_command_extracts_the_marked_span() -> None:
    """(a) Only the marked fence's command is returned, fence markers stripped.

    Also the specific failure of an extractor keyed on backticks or on the word
    ``pyright`` alone: it would return the unrelated ``pytest`` example fence
    above the marker, or a fragment of the begin comment's explanatory prose —
    each a plausible-looking string, so the mistake would not announce itself.
    """
    assert _documented_type_check_command(_HAPPY_DOC) == _HAPPY_COMMAND


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
        (_INVERTED_MARKER_DOC, "inverted"),
        (_MULTILINE_FENCE_DOC, "multi-line fence"),
    ],
)
def test_documented_type_check_command_fails_loudly_on_a_broken_marker(
    markdown_text: str, case: str
) -> None:
    """(b) A broken marker or an unwalkable fence RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns every downstream assertion green while pinning nothing at all.
    Duplicated is the same failure one level down — silently taking the first
    leaves the second mirror unpinned and free to drift. Inverted yields an
    empty slice, which would otherwise extract nothing just as quietly. The
    multi-line fence is the shape-specific case: the chain walker would see only
    the clauses before the line break and pin a strict PREFIX of the chain, so a
    partial mirror would report full coverage. Every message must tell a human
    what to restore and where.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_type_check_command(markdown_text)

    message = str(excinfo.value)
    assert MIRROR_BEGIN in message, case
    assert "CONTRIBUTING.md" in message, case


def test_documented_type_check_command_is_immune_to_the_generic_pyright_decoy() -> None:
    """(c) The generic quality-gate and pre-commit pyright prose is never extracted.

    Pinning either to the live config would be wrong twice over: it would fail
    immediately, and "fixing" it would destroy correct, audience-appropriate
    advice. Both really describe ``hooks/project-checks``'s
    ``PYRIGHT_PACKAGES=(fused-memory orchestrator dashboard)``, which genuinely
    is three packages — the hook is a different gate from the merge-verify chain
    this guard mirrors, and this task deliberately leaves it alone.
    """
    assert _documented_type_check_command(_DECOY_DOC) == _HAPPY_COMMAND


# ---------------------------------------------------------------------------
# The live drift assertion. Everything below reads BOTH committed artifacts
# fresh on every run and never stores its own snapshot of the command — a
# snapshot would just relocate the drift problem into this file.
# ---------------------------------------------------------------------------

_LIVE_LABEL = "the fleet type_check_command (dark-factory-orchestrator.yaml)"
_DOC_LABEL = "the documented Type-check bullet (CONTRIBUTING.md)"


def _fleet_type_check_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["type_check_command"]


def _non_cd_clauses(cmd: str) -> list[list[str]]:
    """Every clause of *cmd* that is not a bare ``cd``, tokenised.

    Used only for the runner-SHAPE assertions below. Splits on the same
    top-level ``&&`` the shared walker uses, so the two views of the chain
    cannot disagree about where a clause begins.
    """
    clauses = []
    for raw in verify_cmd.split_top_level_and(cmd):
        tokens = shlex.split(raw)
        if tokens and tokens[0] == "cd":
            continue
        if tokens:
            clauses.append(tokens)
    return clauses


def test_documented_type_check_bullet_mirrors_the_live_type_check_command() -> None:
    """CONTRIBUTING.md's marked Type-check command must walk the live chain's directories.

    Reads BOTH sides live from the committed artifacts. Compares the package
    DIRECTORIES and their ORDER rather than the command strings, because the two
    lanes diverge at the runner ON PURPOSE (``npx pyright`` for the gate,
    ``uv run pyright`` for a contributor) — see this module's docstring. The
    runner shape is then pinned separately from both ends so that divergence
    cannot quietly collapse.

    MEASURED RED at base HEAD ``6696f1ce0c``: the extractor reported "expected
    exactly one 'type-check-command-mirror:begin' marker, found 0" — CONTRIBUTING.md
    carried no such marker, and its fenced command named 3 of the 7 members the
    gate checks (``cd fused-memory && uv run pyright   # also: orchestrator,
    dashboard``, missing ``shared``, ``escalation``, ``sampler`` and ``cockpit``).
    """
    live_cmd = _fleet_type_check_command()
    documented = _documented_type_check_command(CONTRIBUTING_PATH.read_text(encoding="utf-8"))

    # The gate's own guards read this chain with skip_uv_project=True, so the
    # live side is read exactly as they read it. The DOC side asks the other
    # question — which directories does this command type-check? — for which a
    # `uv run --project X pyright` spelling is a real answer.
    live_cwds = vci.pyright_clause_cwds(live_cmd)
    doc_cwds = vci.pyright_clause_cwds(documented, skip_uv_project=False)

    # (a) NON-VACUITY, both sides. Neither an empty live walk nor an empty
    # documented walk may let this invariant pass by checking nothing at all.
    assert live_cwds, (
        f"{_LIVE_LABEL} walked to no pyright clause at all (task 4108) — this "
        f"mirror invariant would pass vacuously; command: {live_cmd!r}"
    )
    assert doc_cwds, (
        f"{_DOC_LABEL} walked to no pyright clause at all (task 4108) — this "
        f"mirror invariant would pass vacuously; command: {documented!r}"
    )

    # (b) SEMANTIC — ordered equality of the package directories. Ordered rather
    # than set-equal because order is the chain walk's semantics: a mis-tracked
    # relative `cd` produces a wrong-but-same-set result a set comparison passes.
    missing = [d for d in live_cwds if d not in doc_cwds]
    extra = [d for d in doc_cwds if d not in live_cwds]
    assert doc_cwds == live_cwds, (
        f"CONTRIBUTING.md's Type-check bullet has drifted from "
        f"dark-factory-orchestrator.yaml's type_check_command (task 4108).\n"
        f"  MISSING from the doc (type-checked by the gate, not by the "
        f"documented command): {missing}\n"
        f"  EXTRA in the doc (documented but not type-checked by the gate): "
        f"{extra}\n"
        f"  documented: {doc_cwds}\n"
        f"  live:       {live_cwds}\n"
        f"(If both lists hold the same members, the ORDER differs — order is the "
        f"chain walk's semantics, so a reordered `cd ../<member>` hop is a real "
        f"difference.)\n"
        f"UNDER-COVERAGE is the failure this guard exists to stop: a contributor "
        f"who runs the documented command gets a clean LOCAL GREEN over a strict "
        f"subset of what the gate checks, then eats a RED merge verify on a "
        f"branch they believed was ready. If you widened the yaml chain, update "
        f"the command inside the `type-check-command-mirror` marker in "
        f"CONTRIBUTING.md to match."
    )

    # (c) RUNNER SHAPE, pinned from BOTH ends. The divergence is deliberate and
    # documented; pinning only the doc side would let the yaml drift onto the uv
    # lane unnoticed, and pinning neither would let someone "fix" a future red by
    # pasting the yaml's `npx` chain verbatim into CONTRIBUTING.md — silently
    # telling contributors to run the lane the doc itself explains needs `npm ci`
    # first. `test_pyright_version_pin.py` (task 4538) is what holds the two
    # lanes to the same pyright version.
    for tokens in _non_cd_clauses(documented):
        assert tokens == ["uv", "run", "pyright"], (
            f"{_DOC_LABEL} invokes {tokens!r} (task 4108) — every non-`cd` clause "
            f"of the DOCUMENTED chain must be exactly `uv run pyright`, the wheel "
            f"lane a contributor can run without `npm ci`. The gate's `npx "
            f"pyright` (Node lane) is deliberately NOT what the doc tells a "
            f"contributor to run; both resolve the same pinned version, which "
            f"tests/scripts/test_pyright_version_pin.py enforces."
        )
    for tokens in _non_cd_clauses(live_cmd):
        assert tokens == ["npx", "pyright"], (
            f"{_LIVE_LABEL} invokes {tokens!r} (task 4108) — every non-`cd` clause "
            f"of the LIVE chain is expected to be exactly `npx pyright`. If the "
            f"gate deliberately moved to another lane, update this assertion AND "
            f"the CONTRIBUTING.md prose that explains the divergence; do not "
            f"collapse the mirror into a byte copy without deciding that."
        )

    # (d) STALE-TARGET backstop, invisible to (b) once both sides share a typo.
    # A documented directory that does not exist, or that carries no
    # [tool.pyright] table, instructs a contributor to run a command that exits
    # non-zero or that resolves its interpreter from an ambient VIRTUAL_ENV/PATH
    # instead of from that package's own config. (Mirrors assertion (d) of
    # test_contributing_lint_bullet_mirrors_the_live_lint_command.)
    for cwd in doc_cwds:
        package_dir = REPO_ROOT / cwd
        assert package_dir.is_dir(), (
            f"{_DOC_LABEL} names {cwd!r}, which does not exist under {REPO_ROOT} "
            f"(task 4108) — the doc would instruct a contributor to run a command "
            f"that exits non-zero; documented directories: {doc_cwds}"
        )
        pyproject = package_dir / "pyproject.toml"
        assert pyproject.is_file(), (
            f"{_DOC_LABEL} names {cwd!r}, which has no pyproject.toml (task 4108) "
            f"— `uv run pyright` there would resolve neither a project "
            f"environment nor a [tool.pyright] config"
        )
        config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        assert "pyright" in config.get("tool", {}), (
            f"{_DOC_LABEL} names {cwd!r}, whose pyproject.toml declares no "
            f"[tool.pyright] table (task 4108) — the documented command is "
            f"run FROM each package directory precisely so pyright picks up that "
            f"package's own config, including its interpreter pin. Without it "
            f"pyright falls back to the ambient VIRTUAL_ENV/PATH and silently "
            f"type-checks against the wrong interpreter."
        )
