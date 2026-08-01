"""Config-resolution contract: files OUTSIDE a workspace member must resolve the
members' ruff rule set, not ruff's built-in defaults.

Task 3457. ruff resolves its configuration HIERARCHICALLY AND PER FILE, walking
up from each file's directory, and it SKIPS a ``pyproject.toml`` that carries no
``[tool.ruff]`` section. Before this task the repo-root ``pyproject.toml`` had no
such section and no ``ruff.toml`` existed anywhere, so every ``.py`` file outside
a workspace member fell through to ruff's BUILT-IN defaults (``E4``/``E7``/``E9``
+ ``F``) while every member resolved its own declared
``select = ["E", "F", "UP", "B", "SIM", "I"]``.

That asymmetry was live, not theoretical: ``scripts/`` and ``tests/scripts/`` are
both GATED directories (``scripts/orchestrator.yaml`` and
``tests/scripts/orchestrator.yaml`` each declare a directory-wide
``lint_command``, added by tasks 3445 and 3350 respectively), so their lint gates
reported "All checks passed!" while running a rule set nobody chose. Measured at
base ``6357b79b4e``, widening to the members' set surfaced 150 findings — 125 in
``scripts/``, 25 in ``tests/scripts/`` — all of which were FIXED in the same
branch rather than excluded.

MEASURED RED EVIDENCE for the probe below, taken at that same base: the emitted
rule-code sets were ``scripts/`` -> empty, ``tests/scripts/`` -> empty,
``shared/src/shared/`` -> ``{I001, UP017, B905}``. Under the repo-root
``[tool.ruff]`` table all three are equal.

WHY A SUBPROCESS PROBE AND NOT A TOML READ. The claim under test is "which rule
set does ruff RESOLVE for a file at this path", which is a property of ruff's
config discovery, not of any file's contents; a ``tomllib`` read cannot prove it
and an exit code cannot carry it (``ruff check scripts/`` exits 0 both when the
rules are right and when they are absent — that is exactly what the defect
produced). ``--stdin-filename`` makes ruff resolve config as if the fed source
lived at the given path, so the probe is hermetic: NOTHING is written to disk.
Test (b) then pins the DECLARED config so a reader can see the intended values
and a member cannot silently diverge from them.

THIS GUARD MUST NEVER SKIP. ruff is resolved as ``sys.executable -m ruff`` so it
has no PATH dependency, and a missing interpreter or module FAILS rather than
skipping. A guard that skips itself away reproduces the vacuous-pass failure mode
tasks 3350 and 3445 exist to prevent — a green check that never ran.

Production code is cited BY SYMBOL, deliberately never by file:line. The sibling
guard ``test_scripts_module_config.py`` records in its own docstring that its
first draft's line pins were all already stale at HEAD and sent operators to
unrelated code; symbols are greppable and survive edits above them.

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
rather than ``scripts/tests/`` because under FULL_SUITE — a conftest/test-data
trigger, or merge-role ``merge_verify_breadth: full`` — BOTH the ``scripts`` and
``tests/scripts`` module configs run their ``test_command`` VERBATIM and both
target ``tests/scripts/``; the repo-root fleet chain likewise ends in
``pytest tests/scripts/``. A guard living in ``scripts/tests/`` would never run
on merge full-verify.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Violates exactly three rules that are OUTSIDE ruff's built-in defaults but
# INSIDE the members' `select`: I001 (the import block is unsorted), UP017
# (`timezone.utc` -> `datetime.UTC`), B905 (`zip` without explicit `strict=`).
# Under built-in defaults this source is CLEAN, which is what makes it a
# RED/GREEN discriminator rather than a smoke test.
PROBE_SOURCE = '''import sys
import os
from datetime import timezone

X = timezone.utc
Y = list(zip([1], [2]))
print(sys, os, X, Y)
'''

# The two GATED non-member directories this task exists for.
NONMEMBER_PROBE_PATHS = ('scripts/_ruff_probe.py', 'tests/scripts/_ruff_probe.py')

# A live workspace member, used as the BASELINE. Parity is asserted against what
# ruff actually resolves here rather than against a hardcoded rule-code list, so
# this guard keeps holding if the members' `select` is ever changed instead of
# becoming a second source of truth that silently drifts.
BASELINE_PROBE_PATH = 'shared/src/shared/_ruff_probe.py'

# The mechanism, restated once so each failure message can point at it.
_MECHANISM = (
    'ruff resolves config hierarchically PER FILE and SKIPS a pyproject.toml '
    'with no [tool.ruff] section, so with no repo-root table these directories '
    'fall back to ruff built-in defaults (E4/E7/E9 + F) while every workspace '
    'member resolves E,F,UP,B,SIM,I'
)

# Any of these under a non-member directory would shadow the repo-root table and
# silently re-open the asymmetry this task closed.
_SHADOWING_FILENAMES = ('pyproject.toml', 'ruff.toml', '.ruff.toml')

# Directories that MUST stay free of a shadowing config. `scripts/pyproject.toml`
# is doubly forbidden: `orchestrator.config._discover_module_configs` and
# `verify._single_subproject_prefix` both key on pyproject.toml PRESENCE, so
# adding one would silently re-route verify for scripts/ diffs on top of
# shadowing the lint config.
_MUST_NOT_SHADOW = ('scripts', 'tests', 'tests/scripts')


def _resolved_rule_codes(probe_path: str) -> set[str]:
    """Return the rule codes ruff resolves for a file living at *probe_path*.

    Feeds PROBE_SOURCE on stdin with ``--stdin-filename`` so ruff performs its
    real hierarchical config discovery for that path without anything being
    written to disk.
    """
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'ruff',
            'check',
            '--stdin-filename',
            probe_path,
            '--output-format',
            'json',
            '-',
        ],
        input=PROBE_SOURCE,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - only on a broken env
        raise AssertionError(
            f'`{sys.executable} -m ruff check --stdin-filename {probe_path}` did not emit '
            f'parseable JSON (rc={proc.returncode}).\n'
            f'stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}\n'
            'This guard deliberately FAILS rather than skipping when ruff is unavailable: '
            'a guard that skips itself away is the same vacuous-pass failure mode tasks '
            '3350 and 3445 exist to prevent.'
        ) from exc
    return {entry['code'] for entry in payload if entry.get('code')}


def test_nonmember_dirs_resolve_the_member_ruff_rule_set() -> None:
    """scripts/ and tests/scripts/ must resolve the SAME rule set as a member."""
    baseline = _resolved_rule_codes(BASELINE_PROBE_PATH)

    # Non-vacuity: without this, a broken probe passes as the empty set == the
    # empty set, which is precisely the pre-fix state.
    assert baseline, (
        f'The BASELINE probe at {BASELINE_PROBE_PATH} emitted NO rule codes, so the parity '
        'assertions below would pass vacuously as the empty set. Either the probe source no '
        'longer violates I001/UP017/B905, or the workspace member itself lost its '
        f'[tool.ruff] table. {_MECHANISM}.'
    )

    for probe_path in NONMEMBER_PROBE_PATHS:
        resolved = _resolved_rule_codes(probe_path)
        assert resolved == baseline, (
            f'A file at {probe_path} resolves rule codes {sorted(resolved)}, but the '
            f'workspace-member baseline {BASELINE_PROBE_PATH} resolves {sorted(baseline)}.\n'
            f'{_MECHANISM}.\n'
            'Fix: the repo-root pyproject.toml must declare [tool.ruff] / [tool.ruff.lint] '
            'with the same values every workspace member declares (task 3457). This is not '
            'cosmetic: that directory carries a directory-wide lint_command in its '
            'orchestrator.yaml, so the gate reports "All checks passed!" while running a '
            'rule set nobody chose.'
        )


def test_root_ruff_config_matches_every_workspace_member() -> None:
    """The DECLARED root config must equal every member's, and nothing may shadow it."""
    root_toml = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    root_ruff = root_toml.get('tool', {}).get('ruff', {})
    root_lint = root_ruff.get('lint', {})

    assert 'line-length' in root_ruff and 'select' in root_lint and 'ignore' in root_lint, (
        'The repo-root pyproject.toml must declare [tool.ruff].line-length, '
        '[tool.ruff.lint].select and [tool.ruff.lint].ignore (task 3457); found '
        f'[tool.ruff] = {root_ruff!r}.\n{_MECHANISM}.'
    )

    members = root_toml['tool']['uv']['workspace']['members']
    assert members, 'Root [tool.uv.workspace].members is empty; the comparison below is vacuous.'

    # A member that declares NO [tool.ruff] simply inherits the root table, which
    # preserves the invariant — so it is skipped rather than failed. `compared`
    # keeps that leniency from degenerating into a vacuous pass.
    compared = 0
    for member in members:
        member_path = REPO_ROOT / member / 'pyproject.toml'
        assert member_path.is_file(), (
            f'Workspace member {member!r} declared in root [tool.uv.workspace].members has no '
            f'pyproject.toml at {member_path}.'
        )
        member_toml = tomllib.loads(member_path.read_text())
        member_ruff = member_toml.get('tool', {}).get('ruff', {})
        if not member_ruff:
            continue
        member_lint = member_ruff.get('lint', {})
        compared += 1

        if 'line-length' in member_ruff:
            assert member_ruff['line-length'] == root_ruff['line-length'], (
                f'{member}/pyproject.toml declares [tool.ruff].line-length = '
                f'{member_ruff["line-length"]!r}, but the repo root declares '
                f'{root_ruff["line-length"]!r}. Non-member files (scripts/, tests/scripts/, '
                'conftest.py) resolve the ROOT value, so a divergence here means gated '
                'operator tooling is linted differently from the members it drives.'
            )
        for key in ('select', 'ignore'):
            if key in member_lint:
                assert set(member_lint[key]) == set(root_lint[key]), (
                    f'{member}/pyproject.toml declares [tool.ruff.lint].{key} = '
                    f'{sorted(member_lint[key])}, but the repo root declares '
                    f'{sorted(root_lint[key])}. The root table is what every NON-member file '
                    'resolves, so this divergence re-opens the rule-set asymmetry task 3457 '
                    'closed. Compared as sets: ordering is deliberately not pinned.'
                )

    assert compared, (
        'No workspace member declared a [tool.ruff] table, so the equality assertions above '
        'were vacuous. Either the members lost their tables or the member list is wrong.'
    )

    for directory in _MUST_NOT_SHADOW:
        for filename in _SHADOWING_FILENAMES:
            shadow = REPO_ROOT / directory / filename
            assert not shadow.exists(), (
                f'{directory}/{filename} exists and SHADOWS the repo-root [tool.ruff] table — '
                'ruff takes the NEAREST applicable config and does not merge, so this file '
                f'silently decides the rule set for everything under {directory}/.\n'
                'If a per-directory rule set is genuinely wanted, change the repo-root table '
                'and this guard together, deliberately. Note additionally that a '
                'scripts/pyproject.toml is NOT a safe vehicle at all: '
                'orchestrator.config._discover_module_configs and '
                'verify._single_subproject_prefix both key on pyproject.toml PRESENCE, so '
                'adding one silently re-routes verify for scripts/ diffs.'
            )
