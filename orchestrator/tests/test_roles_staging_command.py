"""Behavioral + regression-guard tests for the mandated git staging command.

Task 2745 (refile of 2721): the implementer-mandated staging command
`git add -- . ':!.task'` names the gitignored `.task` path directly in a
pathspec. That trips git's "paths are ignored by one of your .gitignore
files" advice AND makes git exit 1 -- even though staging still succeeds
and correctly excludes `.task/`. Agents kept re-deriving this exit-1 as a
"gotcha" from scratch because the command itself was the friction, not
missing documentation.

The fix drops the redundant `:!.task` exclusion: `.task/` is already
gitignored at the repo root, so a plain `git add -- .` excludes it for
free, with exit 0. `MANDATED_STAGING_COMMAND` is the single source of
truth every staging-rules role prompt must cite, so a partial fix (some
roles missed) can't happen again.

Task 3222 covers a harder sibling of the same 2721/2745 exit-1 case:
untracked `.pytest-tmp/` scratch (from an agent passing
`pytest --basetemp=.pytest-tmp`) can contain *embedded git repos* --
the shape a fixture's own `git init` leaves behind. A commit-less
embedded repo makes `git add -- .` fail HARD with exit 128 (`does not
have a commit checked out` / `fatal: adding files failed`), aborting
the whole staging operation -- worse than the exit-1 warning case. An
embedded repo that DOES have a commit is worse still in a different
way: git exits 0 but silently stages it as a 160000 gitlink that would
then ride the task branch into the merge lane.
"""

from __future__ import annotations

import shlex
import subprocess
import tomllib
from pathlib import Path

from orchestrator.agents.roles import MANDATED_STAGING_COMMAND, ROLES

# The 5 roles whose system_prompt carries a "## CRITICAL: Git Staging Rules"
# section mandating a staging command -- see roles.py.
_STAGING_RULES_ROLES = ('implementer', 'debugger', 'merger', 'steward', 'simple_task')

# Resolved from THIS FILE (orchestrator/tests/ -> repo root), never from the
# process CWD or an ascending `git rev-parse` probe: pytest's --basetemp may
# legally point INSIDE a checkout, and an ascending probe run from there would
# silently retarget the wrong repo. Fixed relative walk only.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROOT_GITIGNORE = _REPO_ROOT / '.gitignore'


def _init_repo_with_gitignored_task_dir(repo: Path) -> None:
    """Create a throwaway repo with a gitignored, populated `.task/` dir.

    Mirrors a real worktree's shape: a repo-root `.gitignore` excluding
    `.task/`, a populated `.task/` (plan.json + iterations.jsonl), and a
    tracked source file with a pending (uncommitted) change to stage.
    """
    subprocess.run(['git', 'init', '-b', 'main'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True, capture_output=True)

    (repo / '.gitignore').write_text('.task/\n')
    src_dir = repo / 'src'
    src_dir.mkdir()
    (src_dir / 'mod.py').write_text("print('initial')\n")
    subprocess.run(
        ['git', 'add', '.gitignore', 'src/mod.py'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ['git', '-c', 'commit.gpgsign=false', 'commit', '-m', 'initial'],
        cwd=repo, check=True, capture_output=True,
    )

    task_dir = repo / '.task'
    task_dir.mkdir()
    (task_dir / 'plan.json').write_text('{"plan": true}\n')
    (task_dir / 'iterations.jsonl').write_text('{"iteration": 1}\n')

    # A real pending change to the tracked source file, so there's something
    # genuine to stage.
    (src_dir / 'mod.py').write_text("print('changed')\n")


def _staged_paths(repo: Path) -> list[str]:
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _init_repo_with_pytest_scratch(
    repo: Path, *, embedded_has_commit: bool, gitignore_text: str | None = None,
) -> None:
    """Create a throwaway repo seeded with real `pytest --basetemp` scratch debris.

    Mirrors `_init_repo_with_gitignored_task_dir`'s shape (git init, a
    tracked `src/mod.py` committed, then a real pending edit) but writes the
    REAL repo-root `.gitignore` verbatim by default -- so callers that rely
    on that default get a genuine test of the artifact task 3222 changes,
    not a synthetic stand-in for it -- and additionally plants
    `.pytest-tmp/test_x0` as an embedded git repo (the shape a fixture's own
    `git init` leaves behind under an agent-supplied `--basetemp=.pytest-tmp`)
    plus a `.pytest_cache/v/cache/lastfailed` file.

    `gitignore_text` overrides the `.gitignore` body written into the repo --
    used by the root-cause test to characterise the UNFIXED state (a
    `.gitignore` lacking the entry) without duplicating this whole fixture.
    Defaults to the real repo-root `.gitignore` when omitted.

    When `embedded_has_commit` is False, the embedded repo is left
    commit-less (a bare `git init`, nothing else) -- this is the shape that
    makes `git add -- .` exit 128. When True, the embedded repo gets its own
    local user config and one commit -- this is the shape that makes
    `git add -- .` exit 0 while silently staging a 160000 gitlink.
    """
    subprocess.run(['git', 'init', '-b', 'main'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True, capture_output=True)

    if gitignore_text is None:
        gitignore_text = _ROOT_GITIGNORE.read_text()
    (repo / '.gitignore').write_text(gitignore_text)
    src_dir = repo / 'src'
    src_dir.mkdir()
    (src_dir / 'mod.py').write_text("print('initial')\n")
    subprocess.run(
        ['git', 'add', '.gitignore', 'src/mod.py'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ['git', '-c', 'commit.gpgsign=false', 'commit', '-m', 'initial'],
        cwd=repo, check=True, capture_output=True,
    )

    # A real pending change to the tracked source file, so there's something
    # genuine to stage.
    (src_dir / 'mod.py').write_text("print('changed')\n")

    # Plant the pytest --basetemp scratch debris.
    scratch_dir = repo / '.pytest-tmp' / 'test_x0'
    scratch_dir.mkdir(parents=True)
    subprocess.run(['git', 'init', '-b', 'main'], cwd=scratch_dir, check=True, capture_output=True)
    if embedded_has_commit:
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'],
            cwd=scratch_dir, check=True, capture_output=True,
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test'], cwd=scratch_dir, check=True, capture_output=True,
        )
        (scratch_dir / 'f.py').write_text("print('scratch')\n")
        subprocess.run(['git', 'add', 'f.py'], cwd=scratch_dir, check=True, capture_output=True)
        subprocess.run(
            ['git', '-c', 'commit.gpgsign=false', 'commit', '-m', 'scratch'],
            cwd=scratch_dir, check=True, capture_output=True,
        )

    cache_dir = repo / '.pytest_cache' / 'v' / 'cache'
    cache_dir.mkdir(parents=True)
    (cache_dir / 'lastfailed').write_text('{}\n')


def test_mandated_command_exits_zero_and_excludes_task_dir(tmp_path: Path) -> None:
    """The mandated command stages tracked changes and excludes `.task/`, exit 0."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_gitignored_task_dir(repo)

    result = subprocess.run(
        shlex.split(MANDATED_STAGING_COMMAND), cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        f'mandated staging command exited non-zero: stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    staged = _staged_paths(repo)
    assert 'src/mod.py' in staged, staged
    assert all(not path.startswith('.task/') for path in staged), staged


def test_legacy_exclusion_form_exits_one_root_cause(tmp_path: Path) -> None:
    """Root-cause-as-spec: the retired `:!.task` exclusion form still exits 1.

    This pins the actual root cause (naming a gitignored path in a
    pathspec) rather than a guess -- if git's behavior here ever changes,
    this test (not just intuition) will catch it.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_gitignored_task_dir(repo)

    result = subprocess.run(
        ['git', 'add', '--', '.', ':!.task'], cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 1, (
        'expected the legacy `:!.task` form to reproduce the exit-1 root cause; got '
        f'returncode={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}'
    )


def test_no_role_system_prompt_mandates_the_legacy_exclusion_form() -> None:
    """No role's system_prompt may mandate the exit-1-prone `:!.task` form.

    Mirrors test_roles_operator_tools.py's convention: iterate ROLES and
    assert an invariant across every role.system_prompt in one place.
    """
    offenders = sorted(name for name, role in ROLES.items() if ':!.task' in role.system_prompt)
    assert offenders == [], (
        f'role(s) still mandate the exit-1-prone `:!.task` staging form: {offenders}'
    )


def test_staging_rules_roles_mandate_the_canonical_constant() -> None:
    """Every staging-rules role's system_prompt cites MANDATED_STAGING_COMMAND
    as its own standalone line -- not merely as a prefix of some other command.

    A bare `MANDATED_STAGING_COMMAND in system_prompt` substring check would
    also pass for the retired `git add -- . ':!.task'` form, since that form
    contains `git add -- .` as a leading substring -- it would not actually
    catch a role that regressed to mandating only the legacy form. Checking
    for an exact, standalone (whitespace-stripped) line gives this test real
    discriminating power on its own, independent of
    test_no_role_system_prompt_mandates_the_legacy_exclusion_form.
    """
    missing = sorted(
        name for name in _STAGING_RULES_ROLES
        if MANDATED_STAGING_COMMAND not in {
            line.strip() for line in ROLES[name].system_prompt.splitlines()
        }
    )
    assert missing == [], (
        f'role(s) missing MANDATED_STAGING_COMMAND as a standalone line in system_prompt: {missing}'
    )


def test_mandated_command_survives_commitless_pytest_basetemp_debris(tmp_path: Path) -> None:
    """The mandated command must not fail on a commit-less pytest-scratch embedded repo.

    A `pytest --basetemp=.pytest-tmp` fixture that does `git init` but never
    commits leaves an embedded repo with NO commit checked out under
    `.pytest-tmp/`. Today, `git add -- .` on that shape fails HARD:
        error: '.pytest-tmp/test_x0/' does not have a commit checked out
        fatal: adding files failed
    exit 128 -- aborting the whole staging operation, so not even the
    genuine pending change to `src/mod.py` gets staged.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_pytest_scratch(repo, embedded_has_commit=False)

    result = subprocess.run(
        shlex.split(MANDATED_STAGING_COMMAND), cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        f'mandated staging command exited non-zero: stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert 'src/mod.py' in _staged_paths(repo)


def test_mandated_command_does_not_stage_pytest_scratch_gitlink(tmp_path: Path) -> None:
    """The mandated command must not silently stage pytest scratch as a gitlink.

    When the embedded repo under `.pytest-tmp/` DOES have a commit, today
    `git add -- .` exits 0 but silently stages it as a 160000 gitlink:
        warning: adding embedded git repository: .pytest-tmp/test_x0
    A gitlink that reaches a task branch would ride into the merge lane.
    This is the quieter and more dangerous sibling of the exit-128 case --
    an exit-code-only assertion would not catch it, which is why this test
    inspects the staged paths instead.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_pytest_scratch(repo, embedded_has_commit=True)

    result = subprocess.run(
        shlex.split(MANDATED_STAGING_COMMAND), cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        f'mandated staging command exited non-zero: stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    staged = _staged_paths(repo)
    assert all(
        not (path.startswith('.pytest-tmp/') or path.startswith('.pytest_cache/'))
        for path in staged
    ), staged


def test_missing_gitignore_entry_reproduces_embedded_repo_exit_128(tmp_path: Path) -> None:
    """Root-cause-as-spec: without the .gitignore entry, git itself exits 128.

    Mirrors test_legacy_exclusion_form_exits_one_root_cause: pins git's
    actual behavior on a commit-less embedded repo under an un-ignored
    `.pytest-tmp/`, so a future git change (not just intuition) would catch
    a regression here. Deliberately writes a minimal `.gitignore` (NOT the
    real root one via `_ROOT_GITIGNORE`) -- this test exists to characterise
    the unfixed state, so it must not depend on whether the real fix has
    landed.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_pytest_scratch(repo, embedded_has_commit=False, gitignore_text='.task/\n')

    result = subprocess.run(
        shlex.split(MANDATED_STAGING_COMMAND), cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 128, (
        'expected a commit-less embedded repo under an un-ignored .pytest-tmp/ to '
        f'reproduce the exit-128 root cause; got returncode={result.returncode} '
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )


def test_no_pyproject_pins_pytest_basetemp_in_addopts() -> None:
    """Guard: no repo pyproject.toml may hard-code --basetemp in addopts.

    pytest exposes no `basetemp` ini key -- `addopts = "--basetemp=..."` is
    the only way to pin one from pyproject.toml. Doing so would be
    destructive: pytest REMOVES the basetemp tree at startup, and
    dark-factory-orchestrator.yaml runs seven subproject suites (several
    with `-n auto`) from different CWDs, so a shared hard-coded basetemp
    would have concurrent/successive suites deleting each other's scratch.
    The fix for pytest scratch debris belongs in .gitignore, not in pinning
    where the debris lands.

    Globs every subproject pyproject.toml (root + one level down, skipping
    hidden dirs like .venv/.worktrees) instead of hard-coding just the root
    and orchestrator files -- the repo has 8 pyproject.toml files today and
    5 of them declare addopts (including fused-memory, which also runs
    `-n auto` like orchestrator does), so a fixed pair would silently stop
    covering the other subprojects the moment one of them added a
    --basetemp pin. Parses the `[tool.pytest.ini_options].addopts` value via
    tomllib rather than scanning the raw file text, so a `--basetemp`
    mention inside a TOML comment can't produce a false positive on an
    otherwise-compliant file.
    """
    candidates = [_REPO_ROOT / 'pyproject.toml'] + sorted(
        path for path in _REPO_ROOT.glob('*/pyproject.toml') if not path.parent.name.startswith('.')
    )
    assert len(candidates) >= 2, f'expected to find multiple repo pyproject.toml files, got {candidates}'

    for candidate in candidates:
        with candidate.open('rb') as f:
            config = tomllib.load(f)
        addopts = config.get('tool', {}).get('pytest', {}).get('ini_options', {}).get('addopts', '')
        assert '--basetemp' not in addopts, (
            f'{candidate} must not pin --basetemp in addopts -- see task 3222 design decision'
        )
