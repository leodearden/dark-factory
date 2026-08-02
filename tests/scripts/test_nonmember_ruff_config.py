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

BOTH HALVES SWEEP, THEY DO NOT ENUMERATE. The probed paths and the forbidden
shadowing configs are DERIVED by walking the repo and subtracting the workspace
members declared in ``[tool.uv.workspace].members``, never listed by hand. A
hand-maintained list is the same failure mode this table closes one level up: it
covers today's two gated directories and lets the NEXT non-member directory —
or a nested ``scripts/legibility/ruff.toml``, which would silently re-scope
roughly half of what ``ruff check scripts/`` covers — re-open the gap without
anything going red. The two gated directories are separately asserted to be
PRESENT in the derived set, so a pruning bug cannot shrink coverage silently.

WHAT TEST (b) COMPARES, AND WHY THE ROOT TABLE MAY DECLARE NOTHING ELSE. Only
``[tool.ruff].line-length`` and ``[tool.ruff.lint].{select,ignore}`` are held
equal to every member. Any OTHER key at the root would be an eighth, unenforced
hand-maintained copy free to drift from the members, so the guard asserts the
root declares none — extend the comparison first, then the table. This is not
hypothetical: ``[tool.ruff.format]`` was declared at the root by this task's
first pass and was ALREADY divergent, since four of the seven members add
``docstring-code-format = true``. It has since been removed; see the
``[tool.ruff]`` comment block in the repo-root ``pyproject.toml``.

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

PROBE_BASENAME = '_ruff_probe.py'

# The two GATED non-member directories this task exists for. These are a FLOOR,
# not the coverage list: the probed set is derived by walking the repo (see
# _nonmember_probe_paths), and these two are asserted to be in it so a pruning
# bug or a moved directory cannot quietly shrink the guard to nothing.
GATED_NONMEMBER_DIRS = ('scripts', 'tests/scripts')

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

# Any of these ANYWHERE outside a workspace member would shadow the repo-root
# table and silently re-open the asymmetry this task closed, because ruff takes
# the NEAREST applicable config and does not merge.
_SHADOWING_FILENAMES = ('pyproject.toml', 'ruff.toml', '.ruff.toml')

# Root [tool.ruff] keys this file COMPARES against every member. Declaring
# anything else at the root creates an unenforced copy that can drift, so the
# root is asserted to declare nothing outside these sets. `lint` is the
# sub-table holding _ROOT_COMPARED_LINT_KEYS.
_ROOT_COMPARED_KEYS = frozenset({'line-length', 'lint'})
_ROOT_COMPARED_LINT_KEYS = frozenset({'select', 'ignore'})

# Never walked: VCS metadata, virtualenvs, caches, build output, and nested task
# worktrees (each is a full checkout of this repo, so descending into one would
# rediscover every member pyproject.toml under a non-member prefix).
_PRUNED_DIR_NAMES = frozenset({
    '.git', '.hg', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.task',
    '.tox', '.venv', '.worktrees', '__pycache__', 'build', 'dist',
    'node_modules', 'venv',
})


def _root_pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())


def _member_roots() -> set[pathlib.Path]:
    """Directories owned by a workspace member, read from the root declaration.

    Members are globbed rather than string-matched because uv permits patterns
    (``packages/*``); ``Path.glob`` handles a literal name identically.
    """
    members = _root_pyproject()['tool']['uv']['workspace']['members']
    roots: set[pathlib.Path] = set()
    for pattern in members:
        roots.update(path for path in REPO_ROOT.glob(pattern) if path.is_dir())
    return roots


def _nonmember_dirs() -> list[pathlib.Path]:
    """Every directory the repo-root [tool.ruff] table governs.

    The whole repo minus the member subtrees minus _PRUNED_DIR_NAMES. Derived,
    never enumerated: a hand-written list covers today's directories and lets the
    next one re-open the gap silently.
    """
    member_roots = _member_roots()
    found: list[pathlib.Path] = []
    stack = [REPO_ROOT]
    while stack:
        directory = stack.pop()
        found.append(directory)
        for child in sorted(directory.iterdir()):
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name in _PRUNED_DIR_NAMES or child.name.endswith('.egg-info'):
                continue
            if child in member_roots:
                continue
            stack.append(child)
    return found


def _nonmember_probe_paths() -> list[str]:
    """One probe path per non-member directory that actually holds Python."""
    return sorted(
        str((directory / PROBE_BASENAME).relative_to(REPO_ROOT))
        for directory in _nonmember_dirs()
        if any(directory.glob('*.py'))
    )


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
    """EVERY non-member directory holding Python must resolve a member's rule set."""
    baseline = _resolved_rule_codes(BASELINE_PROBE_PATH)

    # Non-vacuity: without this, a broken probe passes as the empty set == the
    # empty set, which is precisely the pre-fix state.
    assert baseline, (
        f'The BASELINE probe at {BASELINE_PROBE_PATH} emitted NO rule codes, so the parity '
        'assertions below would pass vacuously as the empty set. Either the probe source no '
        'longer violates I001/UP017/B905, or the workspace member itself lost its '
        f'[tool.ruff] table. {_MECHANISM}.'
    )

    probe_paths = _nonmember_probe_paths()

    # The derived set is the coverage; the two GATED directories are the floor.
    # A prune that swallowed them, or a rename, would otherwise leave this test
    # green while checking nothing that any lint_command targets.
    for gated in GATED_NONMEMBER_DIRS:
        expected = f'{gated}/{PROBE_BASENAME}'
        assert expected in probe_paths, (
            f'{gated}/ carries a directory-wide lint_command ({gated}/orchestrator.yaml) but '
            f'was not discovered by _nonmember_probe_paths(), which found {probe_paths}. '
            'Either the directory moved (update GATED_NONMEMBER_DIRS and its orchestrator.yaml '
            'together), it no longer holds any .py file, or _PRUNED_DIR_NAMES / the member '
            'subtraction is over-pruning — in which case this guard is checking less than it '
            'reads as checking.'
        )

    for probe_path in probe_paths:
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
    root_toml = _root_pyproject()
    root_ruff = root_toml.get('tool', {}).get('ruff', {})
    root_lint = root_ruff.get('lint', {})

    assert 'line-length' in root_ruff and 'select' in root_lint and 'ignore' in root_lint, (
        'The repo-root pyproject.toml must declare [tool.ruff].line-length, '
        '[tool.ruff.lint].select and [tool.ruff.lint].ignore (task 3457); found '
        f'[tool.ruff] = {root_ruff!r}.\n{_MECHANISM}.'
    )

    # ...and NOTHING ELSE. Only the keys asserted equal below are enforced; any
    # other key at the root is a copy nothing holds to the members, free to drift
    # exactly as [tool.ruff.format] already had (four of seven members add
    # `docstring-code-format = true`; the root's first-pass copy did not).
    for scope, declared, allowed in (
        ('[tool.ruff]', set(root_ruff), _ROOT_COMPARED_KEYS),
        ('[tool.ruff.lint]', set(root_lint), _ROOT_COMPARED_LINT_KEYS),
    ):
        assert declared <= allowed, (
            f'The repo-root {scope} declares {sorted(declared - allowed)}, which this guard '
            f'does NOT compare against the workspace members (it compares only '
            f'{sorted(allowed)}). An un-compared root key is an eighth hand-maintained copy '
            'that can silently diverge from the seven members — the invisible-asymmetry class '
            'task 3457 exists to close. Extend the comparison in this test FIRST, then declare '
            'the key; or drop it. See the [tool.ruff] comment block in the repo-root '
            'pyproject.toml for why [tool.ruff.format] in particular is deliberately absent.'
        )

    member_roots = sorted(_member_roots())
    assert member_roots, (
        'Root [tool.uv.workspace].members matched no directory; the comparison below is vacuous.'
    )

    # A member that declares NO [tool.ruff] simply inherits the root table, which
    # preserves the invariant — so it is skipped rather than failed. `compared`
    # keeps that leniency from degenerating into a vacuous pass.
    compared = 0
    for member_root in member_roots:
        member = member_root.relative_to(REPO_ROOT)
        member_path = member_root / 'pyproject.toml'
        assert member_path.is_file(), (
            f'Workspace member {str(member)!r} declared in root [tool.uv.workspace].members has '
            f'no pyproject.toml at {member_path}.'
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

    # SWEPT, not enumerated: every non-member directory, at any depth. A
    # hand-listed set would miss e.g. scripts/legibility/ruff.toml, which
    # silently re-scopes roughly half of what `ruff check scripts/` covers.
    shadows = []
    for directory in _nonmember_dirs():
        for filename in _SHADOWING_FILENAMES:
            candidate = directory / filename
            if not candidate.is_file():
                continue
            # The repo-root pyproject.toml IS the table under test. A root
            # ruff.toml/.ruff.toml is NOT exempt: ruff prefers those over a
            # pyproject.toml in the SAME directory, so one there would replace
            # the very table the assertions above pin.
            if directory == REPO_ROOT and filename == 'pyproject.toml':
                continue
            shadows.append(str(candidate.relative_to(REPO_ROOT)))

    assert not shadows, (
        f'{sorted(shadows)} SHADOW the repo-root [tool.ruff] table — ruff takes the NEAREST '
        'applicable config and does not merge, so each one silently decides the rule set for '
        'everything beneath its directory, at any depth and with no gate going red.\n'
        'If a per-directory rule set is genuinely wanted, change the repo-root table and this '
        'guard together, deliberately. Note additionally that a pyproject.toml under scripts/ '
        'is NOT a safe vehicle at all: orchestrator.config._discover_module_configs and '
        'verify._single_subproject_prefix both key on pyproject.toml PRESENCE, so adding one '
        'silently re-routes verify for scripts/ diffs on top of shadowing the lint config.'
    )
