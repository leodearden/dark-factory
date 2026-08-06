"""Policy guard: this repo runs ``ruff check`` only — ``ruff format`` is NOT adopted.

Task 3441. The decision is recorded in CONTRIBUTING.md section 3 ("Environment"):
this repo lints with ``ruff check`` and deliberately does **not** adopt
``ruff format``. The ``[tool.ruff.format]`` block in each package's
``pyproject.toml`` is retained solely as **ad-hoc style config** — it shapes an
editor-on-save or hand-run ``ruff format`` into a single-quoted, repo-consistent
diff — and is **not** an enforced gate.

WHY A MACHINE CHECK, for a decision that is mostly documentation. The failure
mode this task closes is that SILENCE generates repeat work, and that is
measured rather than predicted: task 3342 tracks an 18-entry duplicate memory
cluster of agents independently rediscovering "ruff format is not gated",
because nothing in CONTRIBUTING.md or CLAUDE.md said so. Prose only reaches
agents who read it; a red test is unmissable.

Three invariants, and they are not the same kind of thing:

  (a) ``test_ruff_format_blocks_are_canonical_across_packages`` — the only one
      RED on arrival. The seven retained blocks had DRIFTED: ``escalation``,
      ``shared`` and ``cockpit`` omitted ``docstring-code-format``, which the
      other four set. Retaining the blocks as style config is only coherent if
      they are IDENTICAL — otherwise an ad-hoc ``ruff format`` on ``shared/``
      behaves differently from one on ``orchestrator/``, which is exactly the
      silent per-package divergence that generates confusion filings.

  (b) ``test_no_lint_gate_invokes_ruff_format`` and
  (c) ``test_ruff_lint_still_ignores_e501`` — GREEN on arrival, riding along in
      a module made red by (a). Their value is PROSPECTIVE: a future adoption
      attempt fails loudly and has to flip policy, docs and this file together,
      instead of drifting into adoption one gate at a time. (c) pins the
      standing ``ignore = ["E501"]`` choice specifically because the formatter
      contradicts it — ``E501`` is tolerated on purpose, and ``ruff format``
      exists to rewrap exactly those lines. If a future task drops that ignore,
      a premise behind the DECLINE has changed and the policy needs re-reading.

WHAT THIS FILE DELIBERATELY DOES NOT DO. It asserts nothing about prose in
CONTRIBUTING.md or CLAUDE.md, and must not be extended to: pinning wording
would go red on any future rewording, which is not a defect. It also asserts
nothing about whether source files ARE format-clean — most are not (as of task
3441, 1124 of 1357 first-party package ``.py`` files would be reformatted), and
that is the expected steady state, not debt.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard actually runs under FULL_SUITE and merge-role
``merge_verify_breadth: full`` — the same reason recorded on
``test_contributing_lint_command_drift.py``.
"""
from __future__ import annotations

import pathlib
import tomllib

import yaml

from orchestrator.config import _discover_module_configs

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The canonical root config filename. Loading it by this exact path means the
# guard fails loudly if the config is renamed, rather than quietly guarding
# nothing.
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

# The pre-commit gate's own check script — the one place outside the yaml
# configs where a formatter leg could plausibly be bolted on.
HOOK_PATH = REPO_ROOT / "hooks" / "project-checks"

# The three command keys that can invoke a tool during verify / pre-commit.
_COMMAND_KEYS = ("lint_command", "test_command", "type_check_command")

# An INVOCATION of the formatter. Matched as a substring of command strings and
# of non-comment hook lines only — never of the repo at large: the bare phrase
# legitimately appears as PROSE elsewhere (fused-memory's config schema and its
# tests carry it as topic-guard cluster phrases), and grepping the tree for it
# would flag those.
_FORMAT_INVOCATION = "ruff format"

# The 4-of-7 majority shape, adopted as canonical by task 3441. Compared as a
# PARSED DICT, never as TOML text: task 3441 step 3 adds a multi-line comment
# directly above each of these blocks, and a text-matching guard would go red on
# its own follow-up step.
CANONICAL_FORMAT_BLOCK: dict[str, object] = {
    "quote-style": "single",
    "indent-style": "space",
    "docstring-code-format": True,
}

# Vendored submodule checkouts. Neither carries a top-level pyproject.toml
# today, so the discovery below already skips them — they are named explicitly
# anyway so that a future submodule sync cannot silently enlarge this guard's
# scope and start asserting repo policy over third-party config.
VENDORED_SUBMODULES = frozenset({"graphiti", "mem0"})

# Attached to (b) and (c), which are regression guards rather than the RED. A
# future adopter who hits them should flip the policy consciously, not delete
# the assertion that noticed.
_ADOPTION_REMEDY = (
    "If you are deliberately adopting `ruff format`, update CONTRIBUTING.md "
    "section 3, CLAUDE.md and this test together — do not delete this "
    "assertion alone."
)


def _first_party_ruff_tables() -> dict[str, dict]:
    """``{package name: parsed [tool.ruff] table}`` for every first-party package.

    DISCOVERED, not hardcoded, so a package added later is covered with no edit
    here. The predicate is "declares a ``[tool.ruff]`` table", which is what
    makes the guard's subject well-defined: the repo-root ``pyproject.toml``
    declares none and drops out by the same predicate rather than by a name
    exclusion.

    Every caller must assert the result is non-empty. That is the vacuity hazard
    documented on ``_documented_lint_command`` in
    ``test_contributing_lint_command_drift.py``: a discovery that silently
    matches nothing turns every downstream assertion green while checking
    nothing at all — strictly worse than no guard, because the check still
    reports success.
    """
    tables: dict[str, dict] = {}
    for pyproject in sorted(REPO_ROOT.glob("*/pyproject.toml")):
        package = pyproject.parent.name
        if package in VENDORED_SUBMODULES:
            continue
        with pyproject.open("rb") as handle:
            parsed = tomllib.load(handle)
        ruff_table = parsed.get("tool", {}).get("ruff")
        if ruff_table is None:
            continue
        tables[package] = ruff_table
    return tables


def _hook_invocation_lines() -> list[tuple[int, str]]:
    """``(1-based line number, text)`` for every non-comment line of the hook.

    Only whole-line comments — first non-whitespace character ``#`` — are
    dropped. A trailing ``# ...`` comment is deliberately NOT stripped: doing
    that correctly needs quote-state tracking, and the two failure directions
    are not symmetric. Missing a real invocation defeats the guard silently;
    flagging a line whose trailing comment merely MENTIONS the formatter is
    loud, self-explaining, and fixed by moving that prose onto its own comment
    line. This errs toward flagging.
    """
    text = HOOK_PATH.read_text(encoding="utf-8")
    return [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_ruff_format_blocks_are_canonical_across_packages() -> None:
    """Every retained ``[tool.ruff.format]`` block must be byte-identical in effect.

    MEASURED RED at base HEAD ``5a7770d239``: ``cockpit``, ``escalation`` and
    ``shared`` each set only ``quote-style``/``indent-style``, while
    ``dashboard``, ``fused-memory``, ``orchestrator`` and ``sampler`` also set
    ``docstring-code-format = true``. Canonicalised to the 4-of-7 majority.

    Under the keep-and-annotate decision these blocks are the repo's ad-hoc
    formatter style config. Divergence between them means the same ad-hoc
    ``ruff format`` produces different results per package — the silent
    inconsistency that makes "is the formatter the standard here?" unanswerable
    by reading the config, which is what this task exists to fix.
    """
    ruff_tables = _first_party_ruff_tables()

    # NON-VACUITY. A glob that matched nothing would make every assertion below
    # pass by iterating an empty dict.
    assert ruff_tables, (
        f"no first-party package under {REPO_ROOT} declares a [tool.ruff] table "
        f"(task 3441) — this policy guard would pass vacuously. Expected the "
        f"packages carrying */pyproject.toml with [tool.ruff]; if the layout "
        f"changed, update the discovery in _first_party_ruff_tables()."
    )

    drift: dict[str, str] = {}
    for package, ruff_table in sorted(ruff_tables.items()):
        block = ruff_table.get("format")
        if block == CANONICAL_FORMAT_BLOCK:
            continue
        if block is None:
            drift[package] = "has no [tool.ruff.format] block at all"
            continue
        missing = {k: v for k, v in CANONICAL_FORMAT_BLOCK.items() if k not in block}
        extra = {k: v for k, v in block.items() if k not in CANONICAL_FORMAT_BLOCK}
        differing = {
            k: f"{block[k]!r} (canonical: {v!r})"
            for k, v in CANONICAL_FORMAT_BLOCK.items()
            if k in block and block[k] != v
        }
        drift[package] = (
            f"missing={missing or '{}'} extra={extra or '{}'} differing={differing or '{}'}"
        )

    assert not drift, (
        "the retained [tool.ruff.format] blocks have drifted apart (task 3441).\n"
        + "".join(f"  {package}/pyproject.toml: {delta}\n" for package, delta in sorted(drift.items()))
        + f"  canonical: {CANONICAL_FORMAT_BLOCK}\n"
        "These blocks are NOT an enforced gate — nothing in hooks/project-checks, "
        "any orchestrator.yaml lint_command, or verify runs `ruff format`. They "
        "are retained as ad-hoc style config so that an editor-on-save or "
        "hand-run format produces a single-quoted, repo-consistent diff. That is "
        "only coherent if they are IDENTICAL: divergence means the same ad-hoc "
        "run reformats one package differently from another. Bring the outlier "
        "into line with the canonical block above; see CONTRIBUTING.md section 3."
    )


def test_no_lint_gate_invokes_ruff_format() -> None:
    """No verify command and no pre-commit hook line may invoke ``ruff format``.

    Covers the root ``dark-factory-orchestrator.yaml`` plus every per-module
    ``orchestrator.yaml`` the orchestrator itself registers, plus
    ``hooks/project-checks``. Together those are the places a formatter leg
    could actually be bolted onto the gate.

    Module configs come from the PRODUCTION walk
    ``config._discover_module_configs`` rather than a hand-rolled glob — the
    precedent set by ``_discover_per_module_configs`` in
    ``test_fallback_verify_config.py`` (task 3350). A ``*/orchestrator.yaml``
    glob looks one level deep and would miss ``tests/scripts/``; a
    ``**/orchestrator.yaml`` glob run from the main checkout would descend
    ``.worktrees/`` and ``.venv/``. Delegating covers exactly the set the
    orchestrator registers, at any depth, with the production pruning.
    """
    root_config = yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))
    commands: dict[str, str] = {
        f"{DF_CONFIG_PATH.name}:{key}": root_config[key]
        for key in _COMMAND_KEYS
        if root_config.get(key)
    }
    for prefix, module_config in sorted(_discover_module_configs(REPO_ROOT).items()):
        for key in _COMMAND_KEYS:
            command = getattr(module_config, key)
            if command:
                commands[f"{prefix}/orchestrator.yaml:{key}"] = command

    # NON-VACUITY, both halves. An empty command set or an unreadable hook must
    # fail loudly rather than let the invariant pass by scanning nothing.
    assert commands, (
        f"discovered no lint/test/type_check commands at all under {REPO_ROOT} "
        f"(task 3441) — this guard would pass vacuously. Expected at least "
        f"{DF_CONFIG_PATH.name}'s own commands."
    )
    hook_lines = _hook_invocation_lines()
    assert hook_lines, (
        f"{HOOK_PATH} has no non-comment lines (task 3441) — this guard would "
        f"pass vacuously; the pre-commit check script should be executable shell."
    )

    offenders = [
        f"{label}: {command!r}"
        for label, command in sorted(commands.items())
        if _FORMAT_INVOCATION in command
    ]
    offenders += [
        f"{HOOK_PATH.relative_to(REPO_ROOT)}:{number}: {line.strip()!r}"
        for number, line in hook_lines
        if _FORMAT_INVOCATION in line
    ]

    assert not offenders, (
        "a verify command or pre-commit hook line invokes `ruff format` "
        "(task 3441), but this repo runs `ruff check` only:\n"
        + "".join(f"  {offender}\n" for offender in offenders)
        + _ADOPTION_REMEDY
        + "\nNote adoption is also more invasive than it looks: a second ruff "
        "sub-command in an && chain lands in verify_cmd's OPAQUE-chain handling, "
        "which keys ToolKind.RUFF on the literal head phrase `ruff check` and "
        "file-scopes the lint leg on it — so this would risk mis-scoping lint "
        "fleet-wide, not just adding a step."
    )


def test_ruff_lint_still_ignores_e501() -> None:
    """Every first-party package must still tolerate long lines (``ignore = ["E501"]``).

    This pins the standing decision the formatter would contradict. ``E501`` is
    ignored ON PURPOSE alongside ``line-length = 100``: the repo tolerates long
    lines, and ``ruff format`` exists to rewrap exactly them. The two tools
    disagree BY DESIGN, which is the first reason adoption was declined.

    So this is not a style assertion — it is a premise check. If a future task
    drops the ignore, the ground under the DECLINE has moved and the policy in
    CONTRIBUTING.md section 3 needs a conscious re-read rather than silent
    inheritance.
    """
    ruff_tables = _first_party_ruff_tables()
    assert ruff_tables, (
        f"no first-party package under {REPO_ROOT} declares a [tool.ruff] table "
        f"(task 3441) — this policy guard would pass vacuously."
    )

    missing = sorted(
        package
        for package, ruff_table in ruff_tables.items()
        if "E501" not in ruff_table.get("lint", {}).get("ignore", [])
    )
    assert not missing, (
        f"these packages no longer ignore E501: {missing} (task 3441). "
        f"`ignore = [\"E501\"]` with `line-length = 100` is the deliberate choice "
        f"to tolerate long lines, and it is the first reason `ruff format` — "
        f"which exists to rewrap them — was declined. Dropping it changes a "
        f"premise of that decision.\n" + _ADOPTION_REMEDY
    )
