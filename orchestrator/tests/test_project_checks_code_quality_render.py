"""pre-commit renders a staged code-quality doc with the staged renderer.

roles.py renders orchestrator/src/orchestrator/agents/code_quality.md at
import, so a doc that cannot render makes every orchestrator raise on its next
restart. A docs-only commit straight to main never meets the merge gate's
orchestrator/tests/test_code_quality.py, so hooks/project-checks runs
scripts/check_staged_code_quality_render.py whenever this commit stages the doc
or its renderer.

Every case builds a throwaway repo from verbatim copies of the real files the
check needs, so it pins the real render() and the real wiring rather than a
restatement of either, and runs them with no virtualenv on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from _orch_helpers import (
    PROBE_HOSTILE_ENV_KEYS,
    assert_isolated_git_repo,
    git_env_with_ceiling,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

DOC = 'orchestrator/src/orchestrator/agents/code_quality.md'
RENDERER = 'orchestrator/src/orchestrator/agents/code_quality.py'
_SCRIPT = 'scripts/check_staged_code_quality_render.py'
_HOOK = 'hooks/project-checks'

_EXECUTABLES = (_HOOK, 'scripts/_checkout.py', _SCRIPT)
_COPIED = (
    *_EXECUTABLES,
    'pyproject.toml',
    'orchestrator/src/orchestrator/__init__.py',
    'orchestrator/src/orchestrator/agents/__init__.py',
    RENDERER,
    DOC,
)

_LAST_RENDERED_HEADING = "'## Do not steer by'"
_RENAMED_HEADING_CONSTANT = "'## Do not steer at all'"


def _bare_env(repo: Path) -> dict[str, str]:
    """git confined to *repo*, and a python3 resolved with no virtualenv in reach."""
    env = git_env_with_ceiling(repo)
    for key in (*PROBE_HOSTILE_ENV_KEYS, 'VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env['PATH'] = os.pathsep.join(
        entry
        for entry in env.get('PATH', '').split(os.pathsep)
        if entry and not (Path(entry).parent / 'pyvenv.cfg').exists()
    )
    return env


def _git(repo: Path, *args: str) -> None:
    assert_isolated_git_repo(repo)
    subprocess.run(
        ['git', *args],
        cwd=repo,
        env=_bare_env(repo),
        capture_output=True,
        text=True,
        check=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(
        ['git', 'init', '--quiet', '-b', 'main'],
        cwd=repo,
        env=_bare_env(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    _git(repo, 'config', 'user.name', 'render check test')
    _git(repo, 'config', 'user.email', 'render-check@example.invalid')
    _git(repo, 'config', 'core.hooksPath', str(repo / '.git' / 'hooks'))
    for relpath in _COPIED:
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / relpath, target)
    for relpath in _EXECUTABLES:
        (repo / relpath).chmod(0o755)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '--quiet', '-m', 'base')
    return repo


def _read(repo: Path, relpath: str) -> str:
    return (repo / relpath).read_text(encoding='utf-8')


def _doc_text(repo: Path) -> str:
    return _read(repo, DOC)


def _write(repo: Path, relpath: str, text: str) -> None:
    (repo / relpath).write_text(text, encoding='utf-8')


def _stage(repo: Path, relpath: str, text: str) -> None:
    _write(repo, relpath, text)
    _git(repo, 'add', '--', relpath)


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f'expected exactly one {old!r} in the copied file'
    return text.replace(old, new)


def _break_heading(doc: str) -> str:
    return _replace_once(doc, '\n## Do not steer by\n', '\n## Do not steer\n')


def _insert_under(doc: str, heading: str, line: str) -> str:
    return _replace_once(doc, f'\n{heading}\n', f'\n{heading}\n\n{line}\n')


def _brace_in(doc: str, heading: str) -> str:
    return _insert_under(doc, heading, 'A line carrying a { brace.')


def _valid_reach_edit(doc: str) -> str:
    return _insert_under(doc, '## Reach', 'One more sentence for an editor.')


def _rename_heading_constant(renderer_source: str) -> str:
    return _replace_once(renderer_source, _LAST_RENDERED_HEADING, _RENAMED_HEADING_CONSTANT)


def _run_script(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['python3', _SCRIPT],
        cwd=repo,
        env=_bare_env(repo),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


_MAIN_ONLY_TIER_REACHED = 'uv stub: the main-only tier was reached'


def _run_hook(repo: Path, branch: str) -> subprocess.CompletedProcess[str]:
    """Run the copied hook with *branch* passed as $2, never left to its default.

    ``uv`` resolves to a stub that fails loudly, so a run that reaches the
    main-only tier's ruff can never pass for a refusal, and never runs a real
    ``uv`` against the throwaway repo.
    """
    stub_bin = repo.parent / 'stub_bin'
    stub_bin.mkdir(exist_ok=True)
    stub_uv = stub_bin / 'uv'
    stub_uv.write_text(f'#!/bin/sh\necho "{_MAIN_ONLY_TIER_REACHED}"\nexit 97\n')
    stub_uv.chmod(0o755)
    env = _bare_env(repo)
    env['PATH'] = f'{stub_bin}{os.pathsep}{env["PATH"]}'
    return subprocess.run(
        [str(repo / _HOOK), str(repo), branch],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


class TestTheScriptRendersTheStagedPair:
    def test_an_unchanged_checkout_passes(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)

        result = _run_script(repo)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_staged_renamed_rendered_heading_is_refused_naming_it(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, DOC, _break_heading(_doc_text(repo)))

        result = _run_script(repo)

        assert result.returncode == 1, result.stdout + result.stderr
        assert _LAST_RENDERED_HEADING in result.stderr

    def test_a_staged_brace_in_a_rendered_section_is_refused_naming_it(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, DOC, _brace_in(_doc_text(repo), '## Two stances'))

        result = _run_script(repo)

        assert result.returncode == 1, result.stdout + result.stderr
        assert "'## Two stances'" in result.stderr

    def test_a_brace_outside_the_rendered_sections_passes(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, DOC, _brace_in(_doc_text(repo), '## Reach'))

        result = _run_script(repo)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_broken_staged_doc_is_refused_though_the_working_tree_is_good(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        original = _doc_text(repo)
        _stage(repo, DOC, _break_heading(original))
        _write(repo, DOC, original)

        result = _run_script(repo)

        assert result.returncode == 1, result.stdout + result.stderr

    def test_a_good_staged_doc_passes_though_the_working_tree_is_broken(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        original = _doc_text(repo)
        _stage(repo, DOC, _valid_reach_edit(original))
        _write(repo, DOC, _break_heading(original))

        result = _run_script(repo)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_staged_renderer_is_judged_with_the_staged_doc(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        original = _read(repo, RENDERER)
        _stage(repo, RENDERER, _rename_heading_constant(original))
        _write(repo, RENDERER, original)

        result = _run_script(repo)

        assert result.returncode == 1, result.stdout + result.stderr
        assert _RENAMED_HEADING_CONSTANT in result.stderr

    def test_a_doc_absent_from_the_index_is_refused_naming_it(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        _git(repo, 'rm', '--cached', '--quiet', '--', DOC)

        result = _run_script(repo)

        assert result.returncode == 1, result.stdout + result.stderr
        assert DOC in result.stderr


class TestTheHookRunsTheRenderCheck:
    @pytest.mark.parametrize('branch', ['main', 'task/6107'])
    def test_a_staged_broken_doc_is_refused_on_every_branch(
        self, tmp_path: Path, branch: str
    ) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, DOC, _break_heading(_doc_text(repo)))

        result = _run_hook(repo, branch)

        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert _LAST_RENDERED_HEADING in output
        assert 'code-quality render check' in result.stdout
        assert 'linting (main branch)' not in output
        assert _MAIN_ONLY_TIER_REACHED not in output

    def test_a_staged_broken_renderer_is_refused_on_a_task_branch(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, RENDERER, _rename_heading_constant(_read(repo, RENDERER)))

        result = _run_hook(repo, 'task/6107')

        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert _RENAMED_HEADING_CONSTANT in output

    def test_a_valid_doc_edit_commits_on_a_task_branch(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, DOC, _valid_reach_edit(_doc_text(repo)))

        result = _run_hook(repo, 'task/6107')

        assert result.returncode == 0, result.stdout + result.stderr
        assert 'code-quality render check' in result.stdout

    def test_an_unrelated_commit_never_spawns_the_render_check(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        _stage(repo, 'README.md', '# hello\n')

        result = _run_hook(repo, 'task/6107')

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert 'code-quality' not in output

    def test_a_staged_deletion_of_the_doc_is_refused_on_a_task_branch(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        _git(repo, 'rm', '--cached', '--quiet', '--', DOC)

        result = _run_hook(repo, 'task/6107')

        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert DOC in output
