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
shadowing configs are DERIVED from ``git ls-files --cached --others
--exclude-standard``, minus the workspace members declared in
``[tool.uv.workspace].members``, never listed by hand. A hand-maintained list is
the same failure mode this table closes one level up: it covers today's two
gated directories and lets the NEXT non-member directory — or a nested
``scripts/legibility/ruff.toml``, which would silently re-scope roughly half of
what ``ruff check scripts/`` covers — re-open the gap without anything going
red. The two gated directories are separately asserted to be PRESENT in the
derived set, so a pruning bug cannot shrink coverage silently.

GIT'S VIEW, NOT THE FILESYSTEM'S — because a sweep is only as good as its
containment, and containment is what makes the derived set identical in a task
worktree and in the ``--recurse-submodules`` clone SETUP.md tells every operator
to make. This file's first draft walked the filesystem and pruned by directory
NAME, which held in a worktree and collapsed anywhere else: measured with that
walk, this worktree yielded 6 probe paths and 0 shadows, while the canonical
``project_root`` clone yielded 2356 probe paths and 369 shadows — foreign
checkouts (``.eval-worktrees`` 218, ``.worktrees-orphaned`` 104, ``.claude``
41) plus the ``graphiti`` and ``mem0`` submodules (4 + 2), each carrying its own
``[tool.ruff]`` and so failing parity against the baseline, and each ``ruff``
subprocess costing ~0.164 s inside a directory whose ``test_command`` is a merge
gate under ``merge_verify_breadth: "full"``. Under the git derivation both
checkouts yield the SAME 6 probe paths and zero shadows. See ``_git_ls_files``
for why each flag is load-bearing. Because that contamination is invisible from
inside a worktree, the containment of ``_nonmember_probe_paths`` and
``_shadowing_configs`` is additionally pinned against a SYNTHETIC repo built
per-test rather than against the ambient checkout — an ambient bound would have
been born green here and proved nothing.

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
import os
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
# not the coverage list: the probed set is derived from git's view of the repo
# (see _nonmember_probe_paths), and these two are asserted to be in it so a
# pruning bug or a moved directory cannot quietly shrink the guard to nothing.
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


def _root_pyproject(root: pathlib.Path = REPO_ROOT) -> dict:
    return tomllib.loads((root / 'pyproject.toml').read_text())


def _member_roots(root: pathlib.Path = REPO_ROOT) -> set[pathlib.Path]:
    """Directories owned by a workspace member, read from the root declaration.

    Members are globbed rather than string-matched because uv permits patterns
    (``packages/*``); ``Path.glob`` handles a literal name identically.
    """
    members = _root_pyproject(root)['tool']['uv']['workspace']['members']
    roots: set[pathlib.Path] = set()
    for pattern in members:
        roots.update(path for path in root.glob(pattern) if path.is_dir())
    return roots


def _git(cwd: pathlib.Path, *args: str) -> str:
    """Run git in *cwd* with a scrubbed environment, FAILING loudly on non-zero.

    ``GIT_*`` is stripped from the environment because pytest can run under a
    hook or wrapper that exports ``GIT_DIR`` / ``GIT_INDEX_FILE`` — this repo's
    own pre-commit hook does — and either would silently redirect these calls at
    a different index than the tree being asked about. Identity and signing are
    pinned inline so nothing depends on the ambient user's git config (they are
    inert for the read-only ``ls-files`` call and load-bearing for the synthetic
    fixture's commits).

    A missing git, or a non-zero exit, raises ``AssertionError`` — it never
    skips. Same stance, and the same reason, as the module docstring gives for
    ruff: a guard that skips itself away is a green check that never ran.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    command = [
        'git',
        '-c', 'user.email=ruff-config-guard@example.invalid',
        '-c', 'user.name=ruff config guard',
        '-c', 'commit.gpgsign=false',
        '-c', 'init.defaultBranch=main',
        *args,
    ]
    try:
        proc = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, env=env, check=False,
        )
    except OSError as exc:  # pragma: no cover - only on a broken env
        raise AssertionError(
            f'Could not execute `git {" ".join(args)}` in {cwd}: {exc!r}. This guard derives '
            'the swept set from git rather than from a filesystem walk, and deliberately FAILS '
            'rather than skipping when git is unavailable.'
        ) from exc
    assert proc.returncode == 0, (
        f'`git {" ".join(args)}` failed in {cwd} with rc={proc.returncode}.\n'
        f'stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}'
    )
    return proc.stdout


def _git_ls_files(root: pathlib.Path, *pathspecs: str) -> list[str]:
    """Repo-relative paths git considers part of the repository at *root*.

    ``--cached --others --exclude-standard`` is the whole containment argument,
    and each half is load-bearing:

    * ``--cached`` alone would miss a NEWLY CREATED, not-yet-committed
      ``scripts/legibility/ruff.toml`` — the live-shadow case the sweep exists
      for — so ``--others`` is added to catch it.
    * ``--exclude-standard`` then drops everything gitignored, which is how the
      foreign checkouts vanish: ``.worktrees/``, ``.worktrees-orphaned/``,
      ``.eval-worktrees/`` and ``.claude/worktrees/`` are all ignored at the repo
      root. Verified to add nothing spurious — with and without ``--others`` the
      tracked ``.py`` count and the shadow list are identical in both checkouts.
    * Submodules need no handling at all: ``graphiti`` and ``mem0`` are single
      mode-160000 gitlink entries, so a ``'*.py'`` pathspec matches nothing
      inside them and no ``.gitmodules`` parsing is required.
    * A nested checkout likewise needs none: git reports a directory holding a
      VALID ``.git`` as one untracked directory entry and never descends into it.

    ``-z`` rather than git's default output, so a path containing a space or a
    quote is not silently mangled by C-style quoting.
    """
    return [
        entry
        for entry in _git(root, 'ls-files', '--cached', '--others', '--exclude-standard',
                          '-z', '--', *pathspecs).split('\0')
        if entry
    ]


def _nonmember_directories(root: pathlib.Path = REPO_ROOT) -> set[str]:
    """Repo-relative directories holding Python that the root table governs.

    Git's view of the repo minus the workspace members. Derived, never
    enumerated: a hand-written list covers today's directories and lets the next
    one re-open the gap silently — and a hand-written PRUNE list, which is what
    this helper used to carry, silently over-collects instead (2356 probe paths
    in the canonical clone against 6 here; see the module docstring).
    """
    members = sorted(path.relative_to(root).as_posix() for path in _member_roots(root))
    directories = set()
    for entry in _git_ls_files(root, '*.py'):
        if any(entry == member or entry.startswith(f'{member}/') for member in members):
            continue
        directories.add(pathlib.PurePosixPath(entry).parent.as_posix())
    return directories


def _nonmember_probe_paths(root: pathlib.Path = REPO_ROOT) -> list[str]:
    """One probe path per non-member directory that actually holds Python."""
    return sorted(
        PROBE_BASENAME if directory == '.' else f'{directory}/{PROBE_BASENAME}'
        for directory in _nonmember_directories(root)
    )


def _shadowing_configs(root: pathlib.Path = REPO_ROOT) -> list[str]:
    """Every non-member config file that would shadow the repo-root table.

    SWEPT, not enumerated: at any depth, so e.g. a ``scripts/legibility/ruff.toml``
    — which would silently re-scope roughly half of what ``ruff check scripts/``
    covers — is caught. The repo-root ``pyproject.toml`` is exempt because it IS
    the table under test; a root ``ruff.toml``/``.ruff.toml`` is NOT, because ruff
    prefers those over a ``pyproject.toml`` in the SAME directory and one there
    would replace the very table this file pins.
    """
    members = sorted(path.relative_to(root).as_posix() for path in _member_roots(root))
    entries = _git_ls_files(root, *(f'*{filename}' for filename in _SHADOWING_FILENAMES))
    found = []
    entries = []
    for entry in entries:
        # git pathspec `*` spans directory separators, which is what makes the
        # sweep depth-independent — but it also makes `*ruff.toml` match e.g.
        # `myruff.toml`, so the basename is re-checked against the real list.
        if pathlib.PurePosixPath(entry).name not in _SHADOWING_FILENAMES:
            continue
        if any(entry == member or entry.startswith(f'{member}/') for member in members):
            continue
        if entry == 'pyproject.toml':
            continue
        found.append(entry)
    return sorted(found)


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
            'together), it no longer holds any .py file, or the git derivation / the member '
            'subtraction is over-pruning — in which case this guard is checking less than it '
            'reads as checking. Note the derived set is what git reports, so a .py file that '
            'is gitignored is deliberately absent: no lint_command would see it either.'
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


# --- Synthetic-repo fixture for the containment guard below --------------------
#
# One directory per hostile class MEASURED in the canonical project_root clone,
# so the guard fails for a named reason rather than on a set diff. Built with a
# real `git init` / `git worktree add` / mode-160000 gitlink rather than mocked,
# because the property under test is what GIT considers part of this repository —
# a mock would just restate the derivation's own assumptions back at it.

_FIXTURE_ROOT_PYPROJECT = """[tool.uv.workspace]
members = ["memberpkg"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "UP", "B", "SIM", "I"]
ignore = ["E501"]
"""

# A foreign checkout's own table, deliberately DIFFERENT from the fixture root's:
# if containment regresses, the leaked file resolves this instead and the parity
# half of this guard would go red too — the graphiti/mem0 shape exactly.
_FIXTURE_FOREIGN_PYPROJECT = '[tool.ruff]\nline-length = 120\n'

# What the derivation must see: the fixture root (it holds root_mod.py) and the
# one genuine tracked non-member directory. Everything else is out of repo.
_FIXTURE_EXPECTED_PROBES = frozenset({PROBE_BASENAME, f'tooling/{PROBE_BASENAME}'})

# Prefixes that must contribute NOTHING, each asserted separately below.
_FIXTURE_FOREIGN_PREFIXES = ('vendored/', 'nested/', '.eval-worktrees/')


def _write(root: pathlib.Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _hostile_fixture_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a throwaway repo holding one directory per measured hostile class."""
    root = tmp_path / 'fixture_repo'
    root.mkdir()
    _git(root, 'init', '-q')

    # In-repo content the derivation MUST see.
    _write(root, 'pyproject.toml', _FIXTURE_ROOT_PYPROJECT)
    _write(root, '.gitignore', '.eval-worktrees/\n')
    _write(root, 'root_mod.py', 'x = 1\n')
    _write(root, 'tooling/tool.py', 'x = 1\n')

    # A workspace member: present and tracked, but SUBTRACTED by _member_roots.
    _write(root, 'memberpkg/pyproject.toml', _FIXTURE_FOREIGN_PYPROJECT)
    _write(root, 'memberpkg/mod.py', 'x = 1\n')

    # CLASS 1 — a gitlink SUBMODULE (the graphiti / mem0 shape). Registered at
    # mode 160000 rather than via `git submodule add`, which needs
    # protocol.file.allow on current git and would make this fixture
    # configuration-dependent.
    _write(root, 'vendored/pyproject.toml', _FIXTURE_FOREIGN_PYPROJECT)
    _write(root, 'vendored/vend.py', 'x = 1\n')
    _git(root / 'vendored', 'init', '-q')
    _git(root / 'vendored', 'add', '-A')
    _git(root / 'vendored', 'commit', '--no-verify', '-q', '-m', 'vendored')
    gitlink_sha = _git(root / 'vendored', 'rev-parse', 'HEAD').strip()

    _git(
        root, 'add', 'pyproject.toml', '.gitignore', 'root_mod.py',
        'tooling/tool.py', 'memberpkg/pyproject.toml', 'memberpkg/mod.py',
    )
    _git(root, 'update-index', '--add', '--cacheinfo', f'160000,{gitlink_sha},vendored')
    _git(root, 'commit', '--no-verify', '-q', '-m', 'fixture')

    # CLASS 2 — a NESTED CHECKOUT: a real `git worktree add`, which is precisely
    # what .worktrees/, .claude/worktrees/ and .worktrees-orphaned/ are. It must
    # be a VALID gitfile: measured, git recurses into a directory whose .git file
    # DANGLES and lists its contents, so a hand-written stub .git file would make
    # this fixture assert something git never does in practice.
    _git(root, 'worktree', 'add', '--detach', '-q', 'nested', 'HEAD')

    # CLASS 3 — a GITIGNORED foreign root, the .eval-worktrees/ shape: the single
    # largest contributor in the canonical clone (218 shadows) and one no
    # directory-name prune list in this file ever named.
    _write(root, '.eval-worktrees/run1/pyproject.toml', _FIXTURE_FOREIGN_PYPROJECT)
    _write(root, '.eval-worktrees/run1/run.py', 'x = 1\n')

    return root


def test_the_derivation_excludes_submodules_and_foreign_checkouts(
    tmp_path: pathlib.Path,
) -> None:
    """The swept set must be GIT's view of this repo, identical in every checkout."""
    root = _hostile_fixture_repo(tmp_path)

    probe_paths = set(_nonmember_probe_paths(root=root))
    shadows = set(_shadowing_configs(root=root))

    # Per-class containment FIRST, so a regression names WHICH class leaked
    # rather than only printing a set diff. Measured against the first-draft
    # iterdir walk, this fixture leaked all three: 7 probe paths and 4 shadows.
    derived = probe_paths | shadows
    for prefix, class_name, evidence in (
        (
            'vendored/',
            'gitlink submodule',
            'graphiti and mem0 are mode-160000 entries in the canonical clone; their contents '
            'are not files of this repository and they declare their own [tool.ruff]',
        ),
        (
            'nested/',
            'nested checkout',
            '.worktrees/, .claude/worktrees/ and .worktrees-orphaned/ are each a FULL checkout '
            'of this repo, so descending into one rediscovers every member pyproject.toml under '
            'a non-member prefix — 104 + 41 shadows measured in the canonical clone',
        ),
        (
            '.eval-worktrees/',
            'gitignored foreign root',
            'the largest single contributor measured in the canonical clone (218 shadows, 1330 '
            'probe paths) and one that no directory-name prune list in this file ever named',
        ),
    ):
        leaked = sorted(path for path in derived if path.startswith(prefix))
        assert not leaked, (
            f'The derivation leaked {leaked} from {prefix} — the {class_name} class. {evidence}. '
            'The swept set must be what `git ls-files --cached --others --exclude-standard` '
            'reports, which excludes submodule contents, nested checkouts and ignored roots '
            'for free; a filesystem walk pruned by directory NAME holds inside a task worktree '
            'and collapses in the canonical clone, where both other tests in this file fail.'
        )

    assert probe_paths == set(_FIXTURE_EXPECTED_PROBES), (
        f'_nonmember_probe_paths derived {sorted(probe_paths)} from the synthetic repo, but the '
        f'only non-member Python this repo actually contains is {sorted(_FIXTURE_EXPECTED_PROBES)} '
        '(the root, holding root_mod.py, and tooling/). Compared as an EXACT set, not a bound: '
        'an over-collecting derivation probes foreign checkouts whose own [tool.ruff] differs '
        'from the baseline, and an under-collecting one silently shrinks this file to nothing. '
        'If the member subtraction regressed instead, memberpkg/ will be in the diff.'
    )

    assert not shadows, (
        f'_shadowing_configs derived {sorted(shadows)} from the synthetic repo, which contains no '
        'shadowing config at all: the fixture root pyproject.toml IS the table under test, '
        'memberpkg/pyproject.toml is a workspace member and is subtracted, and every other one '
        'lives inside the submodule, the nested checkout or the ignored root — none of which is '
        'a file of this repository. Any entry here means the sweep is reporting foreign configs '
        'as live shadows, which makes the guard fail in the canonical clone for a reason no '
        'operator can act on.'
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

    # SWEPT, not enumerated: every non-member directory, at any depth. See
    # _shadowing_configs for what is swept and what is deliberately exempt.
    shadows = _shadowing_configs()

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
