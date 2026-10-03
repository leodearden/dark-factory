"""Policy guard: bare ``file.py:NNN`` citations are TOLERATED DRIFT, not debt.

esc-3815-7 (2026-08-24). The decision is recorded in CONTRIBUTING.md section 2
("Repo layout & conventions") and mirrored in CLAUDE.md: a cross-file reference
in a source comment or docstring is written ``path/to/module.py::symbol``; the
bare line-pinned form is not swept, not linted, and not worth filing a task
over. Fixing a pin is in scope only for prose the change in front of you
introduces or edits.

WHY A MACHINE CHECK, for a decision that is entirely documentation. The same
reason ``test_ruff_format_policy.py`` gives, and it is measured here too rather
than predicted: SILENCE generates repeat work. Before this policy existed the
class had produced ~20 repair tasks, $141.77 of LLM spend, 44 dispatches and 31
escalations (14 of them human-facing) — including FIVE re-promotions of one
measurement-only finding, which is the escalation this guard closes — plus an
8-entry Mem0 ``preferences_and_norms`` cluster of agents independently
rediscovering the very convention CONTRIBUTING.md now states. Prose only reaches
agents who read it; a red test is unmissable.

WHAT WAS MEASURED, stated once here so the numbers are not re-derived per
reader. All at ``1cc7c3f3c6``:

  - 428 bare pins across the four ``src`` trees; 1842 repo-wide in ``.py``.
    The population grew 0 -> 229 comment-pins in five months (+29 in the eight
    days after it was first measured), against a repair lane clearing ~2.2
    tasks/month. Roughly 20:1 against the lane.
  - Hand-adjudicated n=49 (seed 20260824, both files opened per verdict):
    39 wrong / 7 correct / 3 ambiguous => ~80% wrong, Wilson 95% CI 66-89%.
    The earlier ">=25% on n=12" figure counted only MECHANICALLY provable
    defects; those are ~0 here, which is why the two samples disagree.
  - Zero realised harm, with a valid control. 9,579 agent transcripts, 5,629
    escalation records and full ``git log --all`` bodies yield no case of a
    wrong pin misleading a reader; the same method over ``git stash`` returns
    25 real remediation commits and a named incident.

THREE INVARIANTS, and they are not the same kind of thing:

  (a) ``test_policy_is_documented_in_contributing`` and
  (b) ``test_policy_is_mirrored_in_claude_md`` — the pair that carries the
      decision. They assert the marker block still exists and still delimits
      prose, in BOTH homes. The failure they exist for is silent deletion of
      one half: CLAUDE.md is what an agent loads by default and CONTRIBUTING.md
      is what a human is pointed at, so a policy surviving in only one of them
      is a policy half the readership never sees.

  (c) ``test_no_gate_enforces_line_pin_citation_style`` — GREEN on arrival and
      purely PROSPECTIVE. A write-time guard was designed, costed and declined
      (see CONTRIBUTING.md section 2); this assertion makes that a deliberate
      state rather than an oversight, so a future adopter flips policy, docs and
      this file together instead of drifting into enforcement one leg at a time.

WHAT THIS FILE DELIBERATELY DOES NOT DO.

  It asserts NOTHING about the WORDING inside the markers. Pinning prose would
  go red on any future rewording, which is not a defect — the same restraint
  ``test_ruff_format_policy.py`` records.

  It asserts NOTHING about how many pins exist. The population is expected to
  grow: that is what "tolerated drift" means, and a counter here would go red
  for doing exactly what the policy permits.

  It does NOT scan the tree for pins. There is no lint, by choice, and adding
  one through the back door of a test would reverse the decision this file
  exists to record.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard actually runs under FULL_SUITE and merge-role
``merge_verify_breadth: full`` — the same reason recorded on
``test_contributing_lint_command_drift.py`` and ``test_ruff_format_policy.py``.
"""
from __future__ import annotations

import pathlib
import re

import yaml
from orchestrator.config import _discover_module_configs

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The two homes of the policy. Read by exact path, like the sibling guards: a
# rename fails loudly here rather than quietly guarding nothing.
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"
CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"

# The marker pair delimiting the policy prose in both files. Chosen to match the
# existing `lint-command-mirror:begin/end` idiom in CONTRIBUTING.md rather than
# inventing a second convention.
#
# Matched as `<!-- line-pin-policy:begin`, NOT as a whole comment: both homes
# carry explanatory prose inside the opening marker comment itself, and pinning
# the closing `-->` would forbid that.
_MARKER_BEGIN = "<!-- line-pin-policy:begin"
_MARKER_END = "<!-- line-pin-policy:end -->"

# The three command keys that can invoke a checker during verify / pre-commit.
# Same set as test_ruff_format_policy.py — the places an enforcement leg can be
# bolted on.
_COMMAND_KEYS = ("lint_command", "test_command", "type_check_command")

DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

# Every git hook that gates a commit. ALL THREE, for the reason
# test_ruff_format_policy.py records: `pre-commit` runs a universal-guard body
# on every branch before it `exec`s `project-checks`, and `pre-merge-commit` is
# an independent third gate, so a leg added above that `exec` would be real
# enforcement a `project-checks`-only scan never sees.
HOOK_PATHS = (
    REPO_ROOT / "hooks" / "pre-commit",
    REPO_ROOT / "hooks" / "pre-merge-commit",
    REPO_ROOT / "hooks" / "project-checks",
)

# Substrings that would name a line-pin/citation-style checker.
#
# THIS IS A TRIPWIRE ON THE OBVIOUS NAME, NOT A PROOF OF ABSENCE — stated
# plainly so nobody mistakes a green here for "no such guard can exist". A
# determined adopter could name a script `check_prose.py` and slip past. That is
# accepted: the failure this guards against is an adopter wiring enforcement in
# GOOD FAITH without noticing there is a policy to flip, and such a person names
# the script after what it checks.
#
# Deliberately NOT included: the bare word `citation`. `orchestrator.yaml`
# carries it as prose in the comment block above `lint_command`, and matching it
# would flag a comment that merely DESCRIBES the convention.
_ENFORCEMENT_TOKENS = (
    "line_pin",
    "line-pin",
    "pin_style",
    "pin-style",
    "citation_style",
    "citation-style",
)

# Attached to (c), which is a regression guard rather than a red-on-arrival
# finding. A future adopter who hits it should flip the policy consciously.
_ADOPTION_REMEDY = (
    "If you are deliberately adopting a line-pin guard, update CONTRIBUTING.md "
    "section 2, CLAUDE.md and this test together — do not delete this "
    "assertion alone. The design and its costing are in CONTRIBUTING.md "
    "section 2; the decision to decline it was esc-3815-7."
)


def _marked_body(path: pathlib.Path) -> str:
    """Return the prose between the policy markers in *path*.

    Fails the caller's assertion (rather than raising) on every shape that would
    make the guard vacuous: markers missing, duplicated, or inverted. Returning
    an empty string for "present but empty" is the point — a marker pair left
    wrapping nothing is exactly the silent-deletion failure (a) and (b) exist to
    catch, and it must read as absence, not as presence.
    """
    text = path.read_text(encoding="utf-8")
    if text.count(_MARKER_BEGIN) != 1 or text.count(_MARKER_END) != 1:
        return ""
    start = text.index(_MARKER_BEGIN)
    end = text.index(_MARKER_END)
    if end <= start:
        return ""
    # Drop the opening marker comment itself — it carries pointer prose in both
    # homes, which would otherwise satisfy a non-emptiness check on its own.
    body = text[start:end]
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    return body.strip()


def _hook_invocation_lines() -> list[tuple[pathlib.Path, int, str]]:
    """``(path, 1-based line number, text)`` per non-comment line of every gate hook.

    Only whole-line comments are dropped, matching
    ``test_ruff_format_policy.py``: a trailing ``# ...`` is deliberately left in,
    because flagging a line whose comment merely MENTIONS a checker is loud and
    self-explaining, while missing a real invocation defeats the guard silently.
    """
    lines: list[tuple[pathlib.Path, int, str]] = []
    for hook in HOOK_PATHS:
        text = hook.read_text(encoding="utf-8")
        lines += [
            (hook, number, line)
            for number, line in enumerate(text.splitlines(), start=1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return lines


def test_policy_is_documented_in_contributing() -> None:
    """CONTRIBUTING.md section 2 must still carry the marked policy prose.

    This is the human-facing home and the one CLAUDE.md points at. Deleting it
    while leaving the CLAUDE.md mirror in place would leave a pointer to nothing.
    """
    body = _marked_body(CONTRIBUTING_PATH)
    assert body, (
        f"{CONTRIBUTING_PATH.relative_to(REPO_ROOT)} no longer carries exactly one "
        f"well-formed `{_MARKER_BEGIN} ... {_MARKER_END}` block with prose in it "
        f"(esc-3815-7). The bare-`file.py:NNN` tolerated-drift policy lives there "
        f"and is mirrored in {CLAUDE_MD_PATH.name}. If you are reversing the "
        f"policy, update both files and this test together."
    )


def test_policy_is_mirrored_in_claude_md() -> None:
    """CLAUDE.md must still carry the mirror.

    Separate from (a) on purpose. CLAUDE.md is loaded into every session by
    default and CONTRIBUTING.md is not, so this is the copy that actually reaches
    a dispatched agent — the audience whose repeat filings the policy exists to
    stop. A policy surviving only in CONTRIBUTING.md would keep the humans
    informed and change nothing about the task lane.
    """
    body = _marked_body(CLAUDE_MD_PATH)
    assert body, (
        f"{CLAUDE_MD_PATH.relative_to(REPO_ROOT)} no longer carries exactly one "
        f"well-formed `{_MARKER_BEGIN} ... {_MARKER_END}` block with prose in it "
        f"(esc-3815-7). This mirror is what reaches a dispatched agent; "
        f"{CONTRIBUTING_PATH.name} section 2 is the normative copy. Update both "
        f"and this test together."
    )


def test_no_gate_enforces_line_pin_citation_style() -> None:
    """No verify command and no git-hook line may invoke a line-pin checker.

    Covers the root ``dark-factory-orchestrator.yaml``, every per-module
    ``orchestrator.yaml`` the orchestrator registers, and all three gate hooks —
    the places an enforcement leg could actually be bolted on.

    Module configs come from the PRODUCTION walk ``config._discover_module_configs``
    rather than a glob, for the reason ``test_ruff_format_policy.py`` records: a
    one-level glob misses ``tests/scripts/``, and a recursive one run from the
    main checkout descends ``.worktrees/`` and ``.venv/``.

    GREEN ON ARRIVAL. Its value is prospective — see ``_ADOPTION_REMEDY``.
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
        f"(esc-3815-7) — this guard would pass vacuously. Expected at least "
        f"{DF_CONFIG_PATH.name}'s own commands."
    )
    hook_lines = _hook_invocation_lines()
    scanned_hooks = {hook for hook, _, _ in hook_lines}
    assert scanned_hooks == set(HOOK_PATHS), (
        f"scanned no non-comment lines in "
        f"{sorted(str(h.relative_to(REPO_ROOT)) for h in set(HOOK_PATHS) - scanned_hooks)} "
        f"(esc-3815-7) — this guard would pass vacuously over them; every gate "
        f"hook should be executable shell."
    )

    offenders: list[str] = []
    for origin, command in sorted(commands.items()):
        hits = sorted({tok for tok in _ENFORCEMENT_TOKENS if tok in command})
        if hits:
            offenders.append(f"{origin} invokes {hits}: {command}")
    for hook, number, line in hook_lines:
        hits = sorted({tok for tok in _ENFORCEMENT_TOKENS if tok in line})
        if hits:
            offenders.append(
                f"{hook.relative_to(REPO_ROOT)}:{number} invokes {hits}: {line.strip()}"
            )

    assert not offenders, (
        "a gate appears to enforce line-pin citation style, but CONTRIBUTING.md "
        "section 2 records that nothing does, by choice:\n  "
        + "\n  ".join(offenders)
        + f"\n{_ADOPTION_REMEDY}"
    )
