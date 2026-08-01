"""Guard tests: pytest's basetemp can never leak git writes into a live worktree.

Incident esc-3072-3.  Git repository discovery walks UP the directory tree, so
a pytest basetemp nested inside a live task worktree
(``.worktrees/<task>/.pytest-tmp/``) makes every ``cwd=tmp_path`` git call
silently retarget *the enclosing worktree* — three blobs were written into a
real task's object store and ``foo.py`` was staged at stages 1/2/3.

Task 3182 shipped two per-call, opt-in layers in ``_orch_helpers``
(``assert_isolated_git_repo``, ``git_env_with_ceiling``).  This module covers
the two suite-wide, opt-out-impossible layers in the root ``df_pytest_isolation``
module:

* a session-scoped autouse fixture that sets ``GIT_CEILING_DIRECTORIES`` to the
  running basetemp, so the upward walk physically cannot leave it; and
* a collection-time ``--basetemp`` rejection, so a basetemp aimed inside a
  worktree fails loudly before a single test runs.

``_init_repo`` and ``_object_store`` are copied (not imported — ``_orch_helpers``
is not importable from this root-level suite) from
``orchestrator/tests/test_git_repo_isolation_guard.py`` so the containment proof
here is shaped identically to 3182's and the two read as one defence.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    basetemp_rejection_reason,
    git_ceiling_value,
    reject_unsafe_basetemp,
)

_CEILING_KEY = 'GIT_CEILING_DIRECTORIES'


def _init_repo(path: Path) -> Path:
    """``git init`` a fresh repo at *path* with one commit.  Creates its own
    target, so it cannot escape into an enclosing repo.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=path, check=True, capture_output=True,
    )
    (path / 'README.md').write_text('# sentinel\n')
    subprocess.run(['git', 'add', '-A'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-m', 'initial'], cwd=path, check=True, capture_output=True,
    )
    return path


def _object_store(repo: Path) -> set[str]:
    """Snapshot of every loose/packed object file under ``<repo>/.git/objects``.

    Used to prove that a refused call left the victim's object store byte-for-
    byte unchanged — not merely that the command exited non-zero.
    """
    objects = repo / '.git' / 'objects'
    if not objects.is_dir():
        return set()
    return {str(p.relative_to(objects)) for p in objects.rglob('*') if p.is_file()}


def _incident_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Rebuild the esc-3072-3 shape: basetemp nested inside a live worktree.

    Returns ``(victim_repo, simulated_basetemp, nested_non_repo_dir)``.
    """
    victim = _init_repo(tmp_path / 'live-worktree')
    simulated_basetemp = victim / '.pytest-tmp'
    nested = simulated_basetemp / 'test_x0' / 'sub'
    nested.mkdir(parents=True)
    return victim, simulated_basetemp, nested


class TestGitCeilingValueConstruction:
    """git_ceiling_value builds an entry git will actually honour."""

    def test_returns_an_absolute_path_for_a_relative_basetemp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Git silently IGNORES non-absolute ceiling entries.

        A relative entry is not a weaker ceiling, it is *no* ceiling — the walk
        proceeds exactly as if the variable were unset.  Same rationale as
        ``_orch_helpers.git_env_with_ceiling``.
        """
        (tmp_path / 'bt').mkdir()
        monkeypatch.chdir(tmp_path)

        value = git_ceiling_value('bt')

        assert Path(value).is_absolute()
        assert value == str((tmp_path / 'bt').resolve())

    def test_resolves_symlinks(self, tmp_path: Path) -> None:
        """Git compares the ceiling against the RESOLVED cwd.

        An unresolved entry never matches, so the containment is inert.
        """
        real = tmp_path / 'real-basetemp'
        real.mkdir()
        link = tmp_path / 'link-to-basetemp'
        link.symlink_to(real)

        assert git_ceiling_value(link) == str(real.resolve())

    def test_preserves_a_pre_existing_value_and_appends(self, tmp_path: Path) -> None:
        """An operator- or CI-set ceiling must never be clobbered.

        Git treats the variable as a colon-separated list where ANY entry can
        stop the walk, so appending is strictly additive containment.
        """
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()

        value = git_ceiling_value(basetemp, existing='/opt/ci-ceiling')

        assert value == f'/opt/ci-ceiling:{basetemp.resolve()}'

    def test_preserves_every_entry_of_a_multi_entry_value(self, tmp_path: Path) -> None:
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()

        value = git_ceiling_value(basetemp, existing='/opt/a:/opt/b')

        assert value.split(':') == ['/opt/a', '/opt/b', str(basetemp.resolve())]

    def test_is_idempotent_when_the_entry_is_already_present(self, tmp_path: Path) -> None:
        """Conftests nest: a root-rootdir run loads BOTH the root conftest and a
        subproject conftest, so the value must not accumulate duplicates.
        """
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()
        once = git_ceiling_value(basetemp)

        twice = git_ceiling_value(basetemp, existing=once)

        assert twice == once

    def test_is_idempotent_amid_other_entries(self, tmp_path: Path) -> None:
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()
        existing = f'/opt/a:{basetemp.resolve()}:/opt/b'

        assert git_ceiling_value(basetemp, existing=existing) == existing

    def test_an_empty_existing_value_does_not_produce_a_leading_separator(
        self, tmp_path: Path,
    ) -> None:
        """An empty entry is git's "subsequent entries are not symlinks" marker,
        not a no-op — never emit one by accident.
        """
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()

        for empty in (None, ''):
            value = git_ceiling_value(basetemp, existing=empty)
            assert value == str(basetemp.resolve())
            assert '' not in value.split(':')


class TestGitCeilingValueContainsRealGit:
    """The premise proof: the value actually stops real git's upward walk."""

    def test_a_write_from_inside_the_basetemp_cannot_reach_the_enclosing_repo(
        self, tmp_path: Path,
    ) -> None:
        """The esc-3072-3 shape, refused.

        The object store is the assertion that matters — a non-zero exit code
        alone would not prove nothing was written.
        """
        victim, simulated_basetemp, nested = _incident_layout(tmp_path)
        before = _object_store(victim)
        env = os.environ.copy()
        env[_CEILING_KEY] = git_ceiling_value(simulated_basetemp)

        result = subprocess.run(
            ['git', 'hash-object', '-w', '--stdin'],
            cwd=nested, env=env, input=b'leaked payload', capture_output=True,
        )

        assert result.returncode != 0
        assert _object_store(victim) == before

    def test_control_without_the_ceiling_the_write_does_reach_the_repo(
        self, tmp_path: Path,
    ) -> None:
        """Non-vacuity control: the SAME command, ceiling popped, escapes.

        Without this the containment test above could be passing for any
        unrelated reason (a bad git invocation, a missing binary).
        """
        victim, _simulated_basetemp, nested = _incident_layout(tmp_path)
        before = _object_store(victim)
        env = os.environ.copy()
        env.pop(_CEILING_KEY, None)

        result = subprocess.run(
            ['git', 'hash-object', '-w', '--stdin'],
            cwd=nested, env=env, input=b'leaked payload', capture_output=True,
        )

        assert result.returncode == 0, result.stderr.decode()
        assert _object_store(victim) != before

    def test_a_legitimate_repo_under_the_basetemp_still_resolves(
        self, tmp_path: Path,
    ) -> None:
        """No false containment.

        Every legitimate ``tmp_path`` repo and linked worktree lives UNDER the
        basetemp; the ceiling must leave all of them working, or a suite-wide
        ceiling is unshippable.  Git stops only when the walk would ascend INTO
        or above a ceiling entry, so a repo root strictly below it is found
        normally.
        """
        _victim, simulated_basetemp, _nested = _incident_layout(tmp_path)
        inner = _init_repo(simulated_basetemp / 'test_x0' / 'repo')
        deep = inner / 'deep'
        deep.mkdir()
        env = os.environ.copy()
        env[_CEILING_KEY] = git_ceiling_value(simulated_basetemp)

        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=deep, env=env, capture_output=True, text=True,
        )

        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()).resolve() == inner.resolve()


# The gitignored worktree-root vocabulary, .gitignore:15-17.  Spelled out here
# rather than imported so the test pins the values independently of the module.
_WORKTREE_ROOTS = ['.worktrees', '.worktrees-orphaned', '.eval-worktrees']


def _stub_config(basetemp: object) -> SimpleNamespace:
    """A config exposing ONLY ``.option.basetemp``.

    ``reject_unsafe_basetemp`` must read nothing else — ``config._tmp_path_factory``
    is private and its availability during a conftest ``pytest_configure``
    depends on plugin hook ordering.  A stub this thin fails loudly if the
    implementation ever reaches past the public CLI option.
    """
    return SimpleNamespace(option=SimpleNamespace(basetemp=basetemp))


class TestBasetempRejectionReason:
    """A basetemp aimed inside a live worktree is named and explained."""

    @pytest.mark.parametrize('worktree_root', _WORKTREE_ROOTS)
    def test_rejects_every_gitignored_worktree_root(self, worktree_root: str) -> None:
        reason = basetemp_rejection_reason(f'/home/dev/repo/{worktree_root}/3072/.pytest-tmp')

        assert reason

    def test_rejects_the_exact_incident_shape(self) -> None:
        """``.worktrees/3072/.pytest-tmp`` — the basetemp that caused esc-3072-3."""
        reason = basetemp_rejection_reason('/home/leo/src/dark-factory/.worktrees/3072/.pytest-tmp')

        assert reason
        assert '.worktrees' in reason

    def test_the_message_names_the_remedy_not_just_the_verdict(self) -> None:
        """The agent that trips this must be told what to change.

        A bare "unsafe basetemp" verdict costs a round-trip to work out that
        the offending input was a CLI flag.
        """
        reason = basetemp_rejection_reason('/home/dev/repo/.worktrees/3072/.pytest-tmp')

        assert reason is not None
        assert '--basetemp' in reason

    def test_accepts_the_pytest_default_location(self) -> None:
        """The verify lane's case: no --basetemp, so pytest defaults here."""
        assert basetemp_rejection_reason('/tmp/pytest-of-someone/pytest-1') is None

    def test_accepts_a_tmp_path_derived_basetemp(self, tmp_path: Path) -> None:
        assert basetemp_rejection_reason(tmp_path) is None

    def test_accepts_the_repo_root_itself(self) -> None:
        """Only a basetemp INSIDE a worktree root is refused.

        The repo whose .gitignore names these directories is not itself unsafe.
        """
        assert basetemp_rejection_reason('/home/leo/src/dark-factory/build/tmp') is None

    @pytest.mark.parametrize(
        'path',
        [
            '/tmp/my.worktrees-backup/x',
            '/tmp/archived.worktrees/x',
            '/tmp/eval-worktrees-old/x',
        ],
    )
    def test_matches_path_components_not_substrings(self, path: str) -> None:
        """A directory that merely CONTAINS the name is a different directory.

        Substring matching here would refuse unrelated, perfectly safe
        basetemps — turning a safety net into an outage.
        """
        assert basetemp_rejection_reason(path) is None


class TestRejectUnsafeBasetemp:
    """The pytest_configure hook: loud and early, or silent."""

    def test_raises_usage_error_for_an_unsafe_basetemp(self) -> None:
        """UsageError renders as a clean ``ERROR: ...`` with no traceback —
        the right register for an operator/agent misconfiguration.
        """
        config = _stub_config('/home/dev/repo/.worktrees/3072/.pytest-tmp')

        with pytest.raises(pytest.UsageError) as excinfo:
            reject_unsafe_basetemp(config)

        assert '--basetemp' in str(excinfo.value)

    def test_raises_for_a_relative_basetemp_resolved_inside_a_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """How the incident was actually produced: ``--basetemp=.pytest-tmp``
        run from inside the worktree, where the flag value never mentions
        ``.worktrees`` at all.
        """
        worktree = tmp_path / '.worktrees' / '3072'
        worktree.mkdir(parents=True)
        monkeypatch.chdir(worktree)

        with pytest.raises(pytest.UsageError):
            reject_unsafe_basetemp(_stub_config('.pytest-tmp'))

    def test_is_a_no_op_when_no_basetemp_was_passed(self) -> None:
        """The verify lane's case — dark-factory-orchestrator.yaml passes no
        --basetemp, so this must never fire there.
        """
        reject_unsafe_basetemp(_stub_config(None))

    def test_is_a_no_op_for_a_safe_basetemp(self, tmp_path: Path) -> None:
        reject_unsafe_basetemp(_stub_config(tmp_path))


class TestCeilingIsLiveInThisRun:
    """The fixture is WIRED, not merely defined.

    Everything above tests pure functions and a stub config; those would all
    stay green if the fixture were never loaded by any conftest.  This class is
    the only assertion that the defence is actually armed in the process
    running it — the difference between a wired defence and a dead one.
    """

    def test_this_runs_basetemp_is_a_ceiling_entry(
        self, tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Entry membership, not equality: an operator- or CI-set ambient
        ceiling is preserved alongside ours, and under xdist each worker
        contributes its own ``popen-gwN`` basetemp.
        """
        basetemp = str(tmp_path_factory.getbasetemp().resolve())

        entries = os.environ[_CEILING_KEY].split(':')

        assert basetemp in entries, (
            f'{_CEILING_KEY} does not contain this run\'s basetemp {basetemp}.\n'
            'The session ceiling fixture is not loaded for this rootdir. Wire '
            'the test-root conftest to import _df_git_ceiling_at_basetemp from '
            'df_pytest_isolation.'
        )

    def test_the_ambient_ceiling_has_no_duplicate_entries(self) -> None:
        """Idempotence, end-to-end.

        Conftests nest — a run whose rootdir is the repo root loads the root
        conftest AND a subproject conftest — so the pure-function idempotence
        proved above must also hold for the value actually in the environment.
        """
        entries = os.environ[_CEILING_KEY].split(':')

        assert len(entries) == len(set(entries)), f'duplicate entries: {entries}'


# ── Anti-drift: every test-root conftest is wired ─────────────────────────────
#
# The verify lane runs `cd <subproject> && uv run pytest tests/` per segment,
# which makes rootdir the SUBPROJECT — so the repo-root conftest.py is not
# loaded for 7 of the 8 segments.  A defence wired only at the root would be
# dead in exactly the suite (orchestrator, 155 git-invoking test files) that
# needs it most.  This guard is what makes the coverage uniform and keeps it
# that way, rather than trusting the next person to remember.

_WIRING_SNIPPET = (
    'REPO_ROOT = Path(__file__).resolve().parents[2]\n'
    'if str(REPO_ROOT) not in sys.path:\n'
    '    sys.path.append(str(REPO_ROOT))   # APPEND — never insert(0, ...)\n'
    'from df_pytest_isolation import (\n'
    '    _df_git_ceiling_at_basetemp,  # noqa: F401\n'
    '    reject_unsafe_basetemp,\n'
    ')\n'
    'def pytest_configure(config):\n'
    '    reject_unsafe_basetemp(config)\n'
)


def _test_root_conftests() -> list[Path]:
    """Every test-root conftest, enumerated DYNAMICALLY.

    A hard-coded list is how a future subproject silently opts out.  Only ONE
    level is globbed: nested conftests (cockpit/tests/smoke/conftest.py) inherit
    from their test root hierarchically and need no wiring of their own.
    """
    return sorted([REPO_ROOT / 'conftest.py', *REPO_ROOT.glob('*/tests/conftest.py')])


def _file_based_path(node: ast.expr, conftest: Path) -> Path | None:
    """Statically evaluate a ``__file__``-derived path expression, or None.

    Handles the two idioms in use: ``Path(__file__).parent`` (root conftest) and
    ``Path(__file__).resolve().parents[N]`` (subproject conftests).
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'Path':
            if len(node.args) == 1 and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id == '__file__':
                return conftest
            return None
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'resolve':
            base = _file_based_path(node.func.value, conftest)
            return base.resolve() if base is not None else None
        return None
    if isinstance(node, ast.Attribute) and node.attr == 'parent':
        base = _file_based_path(node.value, conftest)
        return base.parent if base is not None else None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
            and node.value.attr == 'parents':
        base = _file_based_path(node.value.value, conftest)
        index = node.slice
        if base is not None and isinstance(index, ast.Constant) and isinstance(index.value, int):
            return base.parents[index.value]
    return None


def _repo_root_names(tree: ast.Module, conftest: Path) -> set[str]:
    """Names bound to a path expression that evaluates to the repo root."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _file_based_path(node.value, conftest)
        if value is not None and value.resolve() == REPO_ROOT:
            names.add(target.id)
    return names


def _sys_path_calls(tree: ast.Module, method: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == 'path'
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == 'sys'
    ]


def _mentions(call: ast.Call, names: set[str]) -> bool:
    return any(
        isinstance(inner, ast.Name) and inner.id in names
        for arg in call.args for inner in ast.walk(arg)
    )


def _wiring_defects(source: str, conftest: Path) -> list[str]:
    """Everything wrong with *source* as a wired test-root conftest.

    Empty list == correctly wired.  Takes the source and its path separately so
    the self-tests below can feed synthetic sources at a real conftest location.
    """
    tree = ast.parse(source)
    defects: list[str] = []

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == 'df_pytest_isolation'
        for alias in node.names
    }
    if '_df_git_ceiling_at_basetemp' not in imported:
        defects.append(
            'does not import _df_git_ceiling_at_basetemp from df_pytest_isolation '
            '(pytest only collects fixtures bound into the conftest namespace, so '
            'the import IS the wiring)'
        )
    if 'reject_unsafe_basetemp' not in imported:
        defects.append('does not import reject_unsafe_basetemp from df_pytest_isolation')

    configures = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'pytest_configure'
    ]
    if not configures:
        defects.append('defines no pytest_configure hook')
    elif not any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == 'reject_unsafe_basetemp'
        for node in configures for inner in ast.walk(node)
    ):
        defects.append('pytest_configure does not call reject_unsafe_basetemp(config)')

    root_names = _repo_root_names(tree, conftest)
    if not root_names:
        defects.append('binds no name to the repo root')
    else:
        if not any(_mentions(c, root_names) for c in _sys_path_calls(tree, 'append')):
            defects.append('does not sys.path.append the repo root')
        if any(_mentions(c, root_names) for c in _sys_path_calls(tree, 'insert')):
            defects.append(
                'sys.path.insert()s the repo root — this makes orchestrator/, shared/, '
                'dashboard/ etc. resolve as namespace packages pointing at the project '
                'folder instead of src/<pkg>/. APPEND so the repo root stays LAST'
            )
    return defects


class TestEveryTestRootConftestIsWired:
    """The defence is uniform across subprojects, and stays that way."""

    def test_all_test_root_conftests_are_wired(self) -> None:
        offenders = {
            str(conftest.relative_to(REPO_ROOT)): defects
            for conftest in _test_root_conftests()
            if (defects := _wiring_defects(conftest.read_text(encoding='utf-8'), conftest))
        }

        assert not offenders, (
            'Unwired test-root conftest(s). The verify lane runs '
            '`cd <subproject> && uv run pytest tests/`, so rootdir is the '
            'SUBPROJECT and the repo-root conftest.py is never loaded — a '
            'conftest missing this wiring runs its whole suite with no basetemp '
            'git ceiling (esc-3072-3).\n'
            f'Offenders: {offenders}\n'
            f'Add:\n{_WIRING_SNIPPET}'
        )

    def test_every_workspace_member_with_tests_has_a_conftest(self) -> None:
        """Closes the "new subproject with no conftest at all" hole.

        The wiring guard above can only inspect files that exist; without this,
        a new member could opt out simply by never adding a conftest.
        """
        pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        members = pyproject['tool']['uv']['workspace']['members']

        missing = [
            m for m in members
            if (REPO_ROOT / m / 'tests').is_dir()
            and not (REPO_ROOT / m / 'tests' / 'conftest.py').exists()
        ]

        assert not missing, (
            f'Workspace member(s) with a tests/ dir but no tests/conftest.py: {missing}. '
            f'The verify lane runs their suites, so they need the wiring too:\n'
            f'{_WIRING_SNIPPET}'
        )

    def test_the_detector_flags_an_unwired_conftest(self) -> None:
        """Self-test: the detector is not vacuously green.

        A structural guard that silently matches nothing is worse than no guard,
        because it reads as coverage.
        """
        sample = 'import sys\nfrom pathlib import Path\n'

        assert _wiring_defects(sample, REPO_ROOT / 'orchestrator' / 'tests' / 'conftest.py')

    def test_the_detector_clears_a_correctly_wired_conftest(self) -> None:
        sample = (
            'import sys\n'
            'from pathlib import Path\n'
            'REPO_ROOT = Path(__file__).resolve().parents[2]\n'
            'if str(REPO_ROOT) not in sys.path:\n'
            '    sys.path.append(str(REPO_ROOT))\n'
            'from df_pytest_isolation import (\n'
            '    _df_git_ceiling_at_basetemp,\n'
            '    reject_unsafe_basetemp,\n'
            ')\n'
            'def pytest_configure(config):\n'
            '    reject_unsafe_basetemp(config)\n'
        )

        assert _wiring_defects(sample, REPO_ROOT / 'orchestrator' / 'tests' / 'conftest.py') == []

    def test_the_detector_flags_an_insert_of_the_repo_root(self) -> None:
        """Self-test: the namespace-package hazard is pinned STRUCTURALLY.

        insert(0, ...) is invisible in review and only bites much later, far
        from this change, as confusing import breakage.
        """
        sample = (
            'import sys\n'
            'from pathlib import Path\n'
            'REPO_ROOT = Path(__file__).resolve().parents[2]\n'
            'sys.path.insert(0, str(REPO_ROOT))\n'
            'from df_pytest_isolation import (\n'
            '    _df_git_ceiling_at_basetemp,\n'
            '    reject_unsafe_basetemp,\n'
            ')\n'
            'def pytest_configure(config):\n'
            '    reject_unsafe_basetemp(config)\n'
        )

        defects = _wiring_defects(sample, REPO_ROOT / 'orchestrator' / 'tests' / 'conftest.py')

        assert any('insert' in d for d in defects)

    def test_the_detector_flags_a_pytest_configure_that_does_not_call_the_check(self) -> None:
        """Self-test: presence of the hook is not enough — the CALL is the point."""
        sample = (
            'import sys\n'
            'from pathlib import Path\n'
            'REPO_ROOT = Path(__file__).resolve().parents[2]\n'
            'sys.path.append(str(REPO_ROOT))\n'
            'from df_pytest_isolation import (\n'
            '    _df_git_ceiling_at_basetemp,\n'
            '    reject_unsafe_basetemp,\n'
            ')\n'
            'def pytest_configure(config):\n'
            "    config.addinivalue_line('markers', 'slow: ...')\n"
        )

        defects = _wiring_defects(sample, REPO_ROOT / 'orchestrator' / 'tests' / 'conftest.py')

        assert any('reject_unsafe_basetemp' in d for d in defects)

    def test_the_root_conftest_is_included_in_the_enumeration(self) -> None:
        """Self-test: the enumeration is not silently empty or root-only."""
        conftests = _test_root_conftests()

        assert REPO_ROOT / 'conftest.py' in conftests
        assert len(conftests) > 1
