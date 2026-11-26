"""`client-feed criteria-results`: a staging run's junit and collection map joined with the criteria file into out/criteria-staging.json."""

import gzip
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _archive_files import junit_xml, passed
from _publisher_pass import reveal_policy
from shared.defect_plugin_location import DEFECT_PLUGIN_DIR

from client_feed.cli import main
from client_feed.contract import CriteriaResults, CriterionOutcome, CriterionResult, RunKind
from client_feed.criteria import STAGING_FILENAME, CriteriaRun, collection_map_path, read_staging
from client_feed.feed_io import OutDir, read_feed_lines
from client_feed.publisher import run_pass
from client_feed.validate import validate_out_dir

CLIENT_FEED = Path(__file__).resolve().parents[1]
EXEMPLAR = Path(__file__).parent / 'fixtures' / 'acceptance' / 'criteria.yaml'
COMMIT = 'b' * 40
PROJECT = 'acme'
B20_CASE = 'tests.test_criteria::test_b20_a_criterion_reads_its_bound_tests'
MARKED = {'tests/test_a.py::test_a': {'defects': [], 'criterion': 'CF-1'}}


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _staging_argv(junit: Path, collection_map: Path, out_dir: Path, **overrides: str) -> list[str]:
    options = {
        '--junit': str(junit),
        '--map': str(collection_map),
        '--criteria': str(EXEMPLAR),
        '--run-kind': 'staging',
        '--commit': COMMIT,
        '--out': str(out_dir),
        **overrides,
    }
    return ['criteria-results', *(part for option in options.items() for part in option)]


def _junit_and_map(tmp_path: Path, name: str = 'r.xml') -> tuple[Path, Path]:
    junit = tmp_path / name
    content = junit_xml(passed('test_a'))
    junit.write_bytes(gzip.compress(content) if name.endswith('.gz') else content)
    collection_map = collection_map_path(junit)
    collection_map.write_text(json.dumps(MARKED))
    return junit, collection_map


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'out'
    path.mkdir()
    return path


@pytest.mark.parametrize('name', ['r.xml', 'r.xml.gz'])
def test_a_staging_run_is_written_and_printed(
    tmp_path: Path, out_dir: Path, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    junit, collection_map = _junit_and_map(tmp_path, name)

    code, out, err = cli(capsys, *_staging_argv(junit, collection_map, out_dir))

    assert code == 0, err
    staged = read_staging(out_dir)
    assert staged is not None
    assert CriteriaRun.model_validate_json(out) == staged
    assert staged.commit == COMMIT
    assert staged.run_at.utcoffset() == timedelta(0)
    assert staged.results == (
        CriterionResult(id='CF-1', outcome=CriterionOutcome.PASSED, tests=('tests.test_a::test_a',)),
        CriterionResult(id='CF-2', outcome=CriterionOutcome.NOT_RUN, tests=()),
    )


def _assert_refused(
    capsys: pytest.CaptureFixture[str], argv: list[str], out_dir: Path, problem: str
) -> None:
    code, out, err = cli(capsys, *argv)

    assert code == 1, out
    assert problem in err
    assert not (out_dir / STAGING_FILENAME).exists()


def test_a_missing_map_is_refused(
    tmp_path: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junit, collection_map = _junit_and_map(tmp_path)
    collection_map.unlink()

    _assert_refused(capsys, _staging_argv(junit, collection_map, out_dir), out_dir, 'unreadable')


@pytest.mark.parametrize(
    'content', [b'format: 2\ncriteria: []\n', b'format: 1\ncriteria: []\n# \xff\xfe\n'],
    ids=['format-2', 'not-utf-8'],
)
def test_invalid_criteria_are_refused(
    tmp_path: Path, out_dir: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    junit, collection_map = _junit_and_map(tmp_path)
    criteria = tmp_path / 'criteria.yaml'
    criteria.write_bytes(content)
    argv = _staging_argv(junit, collection_map, out_dir, **{'--criteria': str(criteria)})

    _assert_refused(capsys, argv, out_dir, 'invalid_criteria')


@pytest.mark.skipif(os.geteuid() == 0, reason='root writes into a read-only directory')
def test_an_out_dir_the_staging_file_cannot_be_written_into_is_refused(
    tmp_path: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junit, collection_map = _junit_and_map(tmp_path)
    out_dir.chmod(0o500)
    try:
        _assert_refused(
            capsys, _staging_argv(junit, collection_map, out_dir), out_dir, 'unwritable'
        )
    finally:
        out_dir.chmod(0o700)


def test_a_malformed_junit_is_refused(
    tmp_path: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junit, collection_map = _junit_and_map(tmp_path)
    junit.write_bytes(b'<testsuites><testsuite>')

    _assert_refused(
        capsys, _staging_argv(junit, collection_map, out_dir), out_dir, 'malformed_junit'
    )


@pytest.mark.parametrize('shape', ['symlink', 'missing'])
def test_an_unusable_out_dir_is_refused(
    tmp_path: Path, out_dir: Path, capsys: pytest.CaptureFixture[str], shape: str
) -> None:
    junit, collection_map = _junit_and_map(tmp_path)
    unusable = tmp_path / 'unusable'
    if shape == 'symlink':
        unusable.symlink_to(out_dir, target_is_directory=True)

    _assert_refused(
        capsys, _staging_argv(junit, collection_map, unusable), out_dir, str(unusable)
    )


@pytest.mark.parametrize(
    'overrides',
    [{'--run-kind': 'verify'}, {'--commit': 'abc123'}],
    ids=['verify-run-kind', 'short-commit'],
)
def test_argparse_refuses_what_a_staging_run_cannot_be(
    tmp_path: Path, out_dir: Path, overrides: dict[str, str]
) -> None:
    junit, collection_map = _junit_and_map(tmp_path)

    with pytest.raises(SystemExit) as caught:
        main(_staging_argv(junit, collection_map, out_dir, **overrides))

    assert caught.value.code == 2
    assert not (out_dir / STAGING_FILENAME).exists()


def _pytest_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS')
    }
    paths = [env['PYTHONPATH']] if env.get('PYTHONPATH') else []
    env['PYTHONPATH'] = os.pathsep.join([*paths, str(DEFECT_PLUGIN_DIR)])
    return env


def test_end_to_end_the_suites_own_marked_test_reaches_the_feed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junit = tmp_path / 'r.xml'
    staged = subprocess.run(
        [
            sys.executable, '-m', 'pytest', 'tests/test_criteria.py', '-m', 'criterion',
            '-p', '_df_defect_plugin', '-p', 'no:cacheprovider', '-q', f'--junitxml={junit}',
        ],
        cwd=CLIENT_FEED,
        env=_pytest_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert staged.returncode == 0, staged.stdout + staged.stderr
    policy = reveal_policy(
        tmp_path, tmp_path / 'in', project_id=PROJECT, committed_at=datetime(2026, 10, 1, tzinfo=UTC)
    )

    code, _, err = cli(
        capsys, *_staging_argv(junit, collection_map_path(junit), policy.out_dir)
    )

    assert code == 0, err
    staging = read_staging(policy.out_dir)
    assert staging is not None
    expected = (
        CriterionResult(id='CF-1', outcome=CriterionOutcome.PASSED, tests=(B20_CASE,)),
        CriterionResult(id='CF-2', outcome=CriterionOutcome.NOT_RUN, tests=()),
    )
    assert staging.results == expected

    run_pass(policy, OutDir.open(policy.out_dir, project_id=PROJECT), now=datetime.now(UTC))

    published = [
        line
        for line in read_feed_lines(policy.out_dir)
        if isinstance(line, CriteriaResults) and line.run_kind is RunKind.STAGING
    ]
    assert [line.results for line in published] == [expected]
    assert validate_out_dir(policy.out_dir).ok
