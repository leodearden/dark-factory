"""The look-around escape hatch's prescribed commands do what the prompt says.

The prompt text is orchestrator/src/orchestrator/agents/harness_guidance.py::_GREP_PCRE_BASH_RECOURSE.
Each command it prescribes is a named template constant spliced into that prose,
so this module runs the templates in a real repo fixture and checks where they
sit in the prompt. It asserts nothing about the surrounding wording.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, _git_env, seed_repo

from orchestrator.agents.harness_guidance import (
    GIT_GREP_PCRE_COMMAND_TEMPLATE,
    GREP_LOOKAROUND_GUIDANCE,
    GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
    GREP_PCRE_FALLBACK_COMMAND_TEMPLATE,
)

_LOOKAHEAD_PATTERN = r'config\.(?!git)[a-z_]+'

_SEARCH_SEED = RepoSeed(
    files=(
        ('.gitignore', 'target/\nnode_modules/\n'),
        ('src/mod.py', 'config.alpha = 1\nconfig.git = 2\n'),
    ),
    message='tracked source plus ignored build and vendored trees',
)


def _render(template: str, pattern: str, path: str) -> list[str]:
    argv = shlex.split(template)
    for placeholder in ('<pattern>', '<path>'):
        assert argv.count(placeholder) == 1, (
            f'expected exactly one {placeholder!r} argv token in the template, got {argv!r}; '
            'rename the placeholder here too if the harness_guidance.py template constant renamed it'
        )
    substitutions = {'<pattern>': pattern, '<path>': path}
    return [substitutions.get(token, token) for token in argv]


def _seed_search_fixture(tmp_path: Path) -> Path:
    repo = seed_repo(tmp_path / 'repo', _SEARCH_SEED)
    (repo / 'src' / 'new_mod.py').write_text('config.beta = 3\n')
    gen = repo / 'target' / 'debug' / 'gen.py'
    gen.parent.mkdir(parents=True)
    gen.write_text('config.gamma = 4\n')
    venv_lib = repo / '.venv' / 'lib'
    venv_lib.mkdir(parents=True)
    (repo / '.venv' / '.gitignore').write_text('*\n')
    (venv_lib / 'site.py').write_text('config.delta = 5\n')
    vendored = repo / 'node_modules' / 'pkg' / 'index.py'
    vendored.parent.mkdir(parents=True)
    vendored.write_text('config.epsilon = 6\n')
    return repo


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd, env=_git_env(cwd), capture_output=True, text=True, check=False,
    )


def _standalone_lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines()}


def _outcome(result: subprocess.CompletedProcess[str]) -> str:
    return f'rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}'


def test_git_grep_template_runs_lookaround_over_tracked_and_untracked_files_only(
    tmp_path: Path,
) -> None:
    repo = _seed_search_fixture(tmp_path)

    result = _run(_render(GIT_GREP_PCRE_COMMAND_TEMPLATE, _LOOKAHEAD_PATTERN, '.'), repo)

    assert result.returncode == 0, _outcome(result)
    matched_paths = {line.split(':', 1)[0] for line in result.stdout.splitlines() if line}
    assert matched_paths == {'src/mod.py', 'src/new_mod.py'}, _outcome(result)
    assert 'config.git' not in result.stdout, _outcome(result)


def test_bash_recourse_cites_the_git_grep_template_as_its_own_line() -> None:
    assert GIT_GREP_PCRE_COMMAND_TEMPLATE in _standalone_lines(GREP_LOOKAROUND_GUIDANCE), (
        'GREP_LOOKAROUND_GUIDANCE must show GIT_GREP_PCRE_COMMAND_TEMPLATE as a '
        'standalone, copyable line, not only inside a longer command'
    )


def test_git_grep_template_refuses_a_directory_outside_any_repo_with_exit_128(
    tmp_path: Path,
) -> None:
    plain = tmp_path / 'plain'
    plain.mkdir()
    (plain / 'mod.py').write_text('config.alpha = 1\n')

    result = _run(_render(GIT_GREP_PCRE_COMMAND_TEMPLATE, _LOOKAHEAD_PATTERN, '.'), plain)

    assert result.returncode == 128, _outcome(result)
    assert 'not a git repository' in result.stderr, _outcome(result)


@pytest.mark.parametrize(
    ('tree', 'seeded_file'),
    [
        ('target', 'target/debug/gen.py'),
        ('.venv', '.venv/lib/site.py'),
        ('node_modules', 'node_modules/pkg/index.py'),
    ],
    ids=['target', 'venv', 'node_modules'],
)
def test_fallback_template_reaches_an_ignored_tree_git_grep_reports_as_no_match(
    tmp_path: Path, tree: str, seeded_file: str,
) -> None:
    repo = _seed_search_fixture(tmp_path)

    git_grep = _run(_render(GIT_GREP_PCRE_COMMAND_TEMPLATE, _LOOKAHEAD_PATTERN, tree), repo)
    assert git_grep.returncode == 1, _outcome(git_grep)
    assert git_grep.stdout == '', _outcome(git_grep)

    fallback_argv = _render(GREP_PCRE_FALLBACK_COMMAND_TEMPLATE, _LOOKAHEAD_PATTERN, tree)
    fallback = _run(['bash', '-c', shlex.join(fallback_argv)], repo)
    assert fallback.returncode == 0, _outcome(fallback)
    assert seeded_file in fallback.stdout, _outcome(fallback)


def test_bash_recourse_leads_with_git_grep_before_the_grep_fallback() -> None:
    standalone_lines = _standalone_lines(GREP_LOOKAROUND_GUIDANCE)
    for name, template in (
        ('GIT_GREP_PCRE_COMMAND_TEMPLATE', GIT_GREP_PCRE_COMMAND_TEMPLATE),
        ('GREP_PCRE_FALLBACK_COMMAND_TEMPLATE', GREP_PCRE_FALLBACK_COMMAND_TEMPLATE),
    ):
        assert template in standalone_lines, (
            f'GREP_LOOKAROUND_GUIDANCE must show {name} as a standalone, copyable line'
        )
    assert GREP_LOOKAROUND_GUIDANCE.index(GIT_GREP_PCRE_COMMAND_TEMPLATE) < (
        GREP_LOOKAROUND_GUIDANCE.index(GREP_PCRE_FALLBACK_COMMAND_TEMPLATE)
    ), 'GREP_LOOKAROUND_GUIDANCE must offer git grep before the command grep -rP fallback'


def test_judge_variant_never_names_the_grep_fallback() -> None:
    assert GREP_PCRE_FALLBACK_COMMAND_TEMPLATE not in GREP_LOOKAROUND_GUIDANCE_READ_ONLY, (
        "GREP_LOOKAROUND_GUIDANCE_READ_ONLY reaches judge, whose Bash(git:*) grant "
        'would refuse GREP_PCRE_FALLBACK_COMMAND_TEMPLATE with a permission denial'
    )
