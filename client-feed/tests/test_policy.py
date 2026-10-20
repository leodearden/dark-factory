"""The publisher policy (plans/client-feed-prd.md §7.6): its data rules, glob semantics, kinds and line counting."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from _git_repo import make_repo
from pydantic import ValidationError

from client_feed.contract import FileKind
from client_feed.policy import Policy, PolicyError, PolicyProblem, load_policy, matches_any


@pytest.fixture
def minimal(tmp_path: Path) -> dict[str, Any]:
    return {
        'project_id': 'acme',
        'project_root': str(tmp_path / 'root'),
        'out_dir': str(tmp_path / 'out'),
        'in_dir': str(tmp_path / 'in'),
        'visible_globs': ['*/src/**', '*/tests/**'],
    }


def _policy(minimal: dict[str, Any], **overrides: Any) -> Policy:
    return Policy.model_validate({**minimal, **overrides})


def test_defaults_follow_the_prd(minimal: dict[str, Any]) -> None:
    policy = _policy(minimal)

    assert policy.interval_seconds == 5
    assert policy.reveal_proportion == 0.25
    assert policy.reveal_floor_lines == 50
    assert policy.max_line_bytes == 400
    assert policy.max_cases_per_run == 5000
    assert policy.test_globs == ('**/tests/**', '**/test_*.py', '**/*_test.py', '**/conftest.py')
    assert policy.docs_globs == ('**/*.md', 'docs/**')
    assert policy.never_reveal_globs == ()
    assert policy.generated_globs == ()
    assert policy.free_globs == policy.test_globs + policy.docs_globs


def test_free_globs_default_to_the_test_and_docs_globs_as_given(minimal: dict[str, Any]) -> None:
    policy = _policy(minimal, test_globs=['spec/**'])

    assert policy.free_globs == ('spec/**', '**/*.md', 'docs/**')


def test_an_explicit_free_globs_is_kept_verbatim(minimal: dict[str, Any]) -> None:
    policy = _policy(minimal, test_globs=['spec/**'], free_globs=['notes/**'])

    assert policy.free_globs == ('notes/**',)


def test_visible_globs_is_required(minimal: dict[str, Any]) -> None:
    del minimal['visible_globs']

    with pytest.raises(ValidationError, match='visible_globs'):
        Policy.model_validate(minimal)


def test_visible_globs_must_not_be_empty(minimal: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='visible_globs'):
        _policy(minimal, visible_globs=[])


def test_an_unknown_key_is_refused(minimal: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='visible_glob'):
        _policy(minimal, visible_glob=['src/**'])


def test_the_policy_is_frozen(minimal: dict[str, Any]) -> None:
    policy = _policy(minimal)
    field = 'interval_seconds'

    with pytest.raises(ValidationError):
        setattr(policy, field, 9)


@pytest.mark.parametrize('field', ['project_root', 'out_dir', 'in_dir'])
def test_a_relative_path_is_refused(minimal: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        _policy(minimal, **{field: 'relative/dir'})


def test_out_dir_and_in_dir_must_differ(minimal: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='in_dir'):
        _policy(minimal, in_dir=minimal['out_dir'])


@pytest.mark.parametrize('field', ['out_dir', 'in_dir'])
def test_out_and_in_dirs_must_lie_outside_the_project_root(
    minimal: dict[str, Any], field: str
) -> None:
    inside = str(Path(minimal['project_root']) / 'publish')

    with pytest.raises(ValidationError) as caught:
        _policy(minimal, **{field: inside})

    assert inside in str(caught.value)
    assert minimal['project_root'] in str(caught.value)


@pytest.mark.parametrize('field', ['out_dir', 'in_dir'])
def test_out_and_in_dirs_must_not_be_the_project_root(
    minimal: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        _policy(minimal, **{field: minimal['project_root']})


@pytest.mark.parametrize('glob', ['', '/abs/**', 'a/../b', './x', 'a//b'])
@pytest.mark.parametrize('field', ['visible_globs', 'never_reveal_globs', 'test_globs'])
def test_a_malformed_glob_is_refused(minimal: dict[str, Any], field: str, glob: str) -> None:
    with pytest.raises(ValidationError, match=field):
        _policy(minimal, **{field: [glob]})


@pytest.mark.parametrize(
    ('glob', 'path', 'expected'),
    [
        ('*/src/**', 'a/src/x.py', True),
        ('*/src/**', 'a/src/b/c.py', True),
        ('*/src/**', 'src/x.py', False),
        ('*/src/**', 'a/b/src/x.py', False),
        ('**/tests/**', 'tests/x.py', True),
        ('**/tests/**', 'a/tests/b/c.py', True),
        ('**/tests/**', 'a/tests', False),
        ('**/*.md', 'README.md', True),
        ('**/*.md', 'a/b.md', True),
        ('**/*.md', 'a/b.mdx', False),
        ('docs/**', 'docs/x/y.md', True),
        ('docs/**', 'a/docs/x', False),
        ('*.py', 'a.py', True),
        ('*.py', 'a/b.py', False),
        ('a?c', 'abc', True),
        ('a?c', 'a/c', False),
        ('[ab].py', 'a.py', True),
        ('[ab].py', 'b.py', True),
        ('[ab].py', 'c.py', False),
        ('[!ab].py', 'c.py', True),
        ('[!ab].py', 'a.py', False),
        ('a.b', 'aXb', False),
        ('src/**/gen/*.py', 'src/gen/x.py', True),
        ('src/**/gen/*.py', 'src/a/b/gen/x.py', True),
        ('src/**/gen/*.py', 'src/a/gen/b/x.py', False),
    ],
)
def test_glob_semantics(glob: str, path: str, expected: bool) -> None:
    assert matches_any([glob], path) is expected


def test_matches_any_is_true_when_any_glob_matches() -> None:
    assert matches_any(['docs/**', '*/src/**'], 'a/src/x.py')
    assert not matches_any(['docs/**', '*/src/**'], 'src/x.py')
    assert not matches_any([], 'src/x.py')


def test_is_visible_reads_the_visible_globs(minimal: dict[str, Any]) -> None:
    policy = _policy(minimal)

    assert policy.is_visible('pkg/src/a.py')
    assert policy.is_visible('pkg/tests/test_a.py')
    assert not policy.is_visible('dark-factory-orchestrator.yaml')
    assert not policy.is_visible('pkg/README.md')


@pytest.mark.parametrize(
    ('path', 'linguist_generated', 'kind'),
    [
        ('pkg/src/a.py', False, FileKind.PRODUCTION),
        ('pkg/src/a.py', True, FileKind.GENERATED),
        ('pkg/tests/test_a.py', True, FileKind.GENERATED),
        ('pkg/build/out.py', False, FileKind.GENERATED),
        ('pkg/tests/build/out.py', False, FileKind.GENERATED),
        ('pkg/tests/helper.py', False, FileKind.TEST),
        ('pkg/src/test_x.py', False, FileKind.TEST),
        ('pkg/tests/README.md', False, FileKind.TEST),
        ('pkg/src/README.md', False, FileKind.DOCS),
        ('docs/guide.txt', False, FileKind.DOCS),
    ],
)
def test_kind_of_ranks_generated_then_test_then_docs(
    minimal: dict[str, Any], path: str, linguist_generated: bool, kind: FileKind
) -> None:
    policy = _policy(minimal, generated_globs=['**/build/**'])

    assert policy.kind_of(path, linguist_generated=linguist_generated) is kind


@pytest.mark.parametrize(
    ('content', 'lines'),
    [
        (b'', 0),
        (b'\n', 0),
        (b'   \n\t\n', 0),
        (b'a\n', 1),
        (b'a', 1),
        (b'a\nb', 2),
        (b'a\n\nb\n', 2),
        (b'x' * 10 + b'\n', 1),
        (b'x' * 11 + b'\n', 2),
        (b'x' * 30, 3),
        (b'x' * 31, 4),
        (b'  ' + b'x' * 9 + b'\n', 2),
    ],
)
def test_counted_lines_is_the_prd_counting_rule(
    minimal: dict[str, Any], content: bytes, lines: int
) -> None:
    policy = _policy(minimal, max_line_bytes=10)

    assert policy.counted_lines(content) == lines


@pytest.fixture
def on_disk(minimal: dict[str, Any]) -> dict[str, Any]:
    """`minimal` made real: project_root a repository with one commit, out_dir an existing directory."""
    repo = make_repo(Path(minimal['project_root']))
    repo.write('src/a.py', 'a = 1\n')
    repo.commit('one', when=datetime(2026, 1, 1, tzinfo=UTC))
    Path(minimal['out_dir']).mkdir()
    return minimal


def _config(tmp_path: Path, mapping: dict[str, Any]) -> Path:
    path = tmp_path / 'policy.yaml'
    path.write_text(yaml.safe_dump(mapping))
    return path


def _refusal(config: Path) -> PolicyError:
    with pytest.raises(PolicyError) as caught:
        load_policy(config)
    assert caught.value.config == config
    return caught.value


def test_a_valid_config_loads_as_its_policy(tmp_path: Path, on_disk: dict[str, Any]) -> None:
    assert load_policy(_config(tmp_path, on_disk)) == Policy.model_validate(on_disk)


def test_a_missing_config_is_unreadable(tmp_path: Path) -> None:
    config = tmp_path / 'absent.yaml'

    refusal = _refusal(config)

    assert refusal.problem is PolicyProblem.UNREADABLE
    assert str(config) in str(refusal)


def test_a_config_that_is_not_text_is_unreadable(tmp_path: Path) -> None:
    config = tmp_path / 'policy.yaml'
    config.write_bytes(b'project_id: \xff\xfe\n')

    assert _refusal(config).problem is PolicyProblem.UNREADABLE


@pytest.mark.parametrize('text', ['visible_globs: [src/**\n', '- a\n- b\n', 'just text\n', ''])
def test_a_config_that_is_not_a_yaml_mapping_is_invalid(tmp_path: Path, text: str) -> None:
    config = tmp_path / 'policy.yaml'
    config.write_text(text)

    assert _refusal(config).problem is PolicyProblem.INVALID


def test_a_schema_violation_is_invalid_and_names_the_field(
    tmp_path: Path, on_disk: dict[str, Any]
) -> None:
    del on_disk['visible_globs']

    refusal = _refusal(_config(tmp_path, on_disk))

    assert refusal.problem is PolicyProblem.INVALID
    assert 'visible_globs' in refusal.detail
    assert 'visible_globs' in str(refusal)


def test_a_root_that_is_no_repository_is_refused(
    tmp_path: Path, minimal: dict[str, Any]
) -> None:
    Path(minimal['project_root']).mkdir()
    Path(minimal['out_dir']).mkdir()

    refusal = _refusal(_config(tmp_path, minimal))

    assert refusal.problem is PolicyProblem.NOT_A_REPOSITORY_ROOT
    assert minimal['project_root'] in str(refusal)
    assert 'not a usable git work tree' in refusal.detail


def test_a_root_inside_a_repository_but_not_its_top_is_refused(
    tmp_path: Path, on_disk: dict[str, Any]
) -> None:
    subdirectory = str(Path(on_disk['project_root']) / 'src')

    refusal = _refusal(_config(tmp_path, {**on_disk, 'project_root': subdirectory}))

    assert refusal.problem is PolicyProblem.NOT_A_REPOSITORY_ROOT
    assert subdirectory in str(refusal)


def test_a_missing_out_dir_is_unusable(tmp_path: Path, on_disk: dict[str, Any]) -> None:
    Path(on_disk['out_dir']).rmdir()

    refusal = _refusal(_config(tmp_path, on_disk))

    assert refusal.problem is PolicyProblem.OUT_DIR_UNUSABLE
    assert on_disk['out_dir'] in str(refusal)


def test_an_out_dir_that_is_a_file_is_unusable(tmp_path: Path, on_disk: dict[str, Any]) -> None:
    Path(on_disk['out_dir']).rmdir()
    Path(on_disk['out_dir']).write_text('not a directory')

    assert _refusal(_config(tmp_path, on_disk)).problem is PolicyProblem.OUT_DIR_UNUSABLE


def test_an_out_dir_that_is_a_symlink_to_a_directory_is_unusable(
    tmp_path: Path, on_disk: dict[str, Any]
) -> None:
    real = tmp_path / 'real-out'
    real.mkdir()
    Path(on_disk['out_dir']).rmdir()
    Path(on_disk['out_dir']).symlink_to(real)

    assert _refusal(_config(tmp_path, on_disk)).problem is PolicyProblem.OUT_DIR_UNUSABLE


@pytest.mark.skipif(os.geteuid() == 0, reason='root owns every directory it could be refused')
def test_an_out_dir_owned_by_another_user_is_refused(
    tmp_path: Path, on_disk: dict[str, Any]
) -> None:
    refusal = _refusal(_config(tmp_path, {**on_disk, 'out_dir': '/'}))

    assert refusal.problem is PolicyProblem.OUT_DIR_NOT_OWNED
    assert 'out_dir / ' in str(refusal)
