"""suite_census.main composes Part 1 and Part 2 into one dated report, or exits 2 naming why not."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest
import suite_census
import suite_census_evidence as ev
import suite_census_outcomes as oc
import suite_census_pinning as pinning
import suite_census_rust as rust
from suite_census_fixtures import nextest_project, pytest_project


def _argv(ecosystem: str, project: Path, out: Path, *extra: str) -> list[str]:
    return [
        '--ecosystem', ecosystem, '--root', str(project), '--tree', str(project),
        '--project', 'demo', '--date', '2026-10-04', '--out', str(out), *extra,
    ]


def _head(tree: Path) -> str:
    return subprocess.run(
        ['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def pytest_tree(tmp_path: Path) -> Path:
    return pytest_project(tmp_path / 'project')


class TestPytestReport:
    def test_header_command_and_both_parts(self, pytest_tree, tmp_path, capsys):
        out = tmp_path / 'report.md'
        argv = _argv('pytest', pytest_tree, out)
        assert suite_census.main(argv) == 0
        text = out.read_text()
        assert 'demo' in text and '2026-10-04' in text and _head(pytest_tree) in text
        assert shlex.join(['python', 'scripts/suite_census.py', *argv]) in text
        part_1 = oc.render_markdown(
            oc.census_outcomes(ev.pytest_evidence(pytest_tree, pytest_tree)), top=100,
        )
        assert part_1 in text
        assert pinning.render_markdown(pinning.measure_python_tree(pytest_tree)) in text
        assert str(out) in capsys.readouterr().out

    def test_top_truncates_the_ranking(self, pytest_tree, tmp_path):
        out = tmp_path / 'report.md'
        assert suite_census.main(_argv('pytest', pytest_tree, out, '--top', '1')) == 0
        census = oc.census_outcomes(ev.pytest_evidence(pytest_tree, pytest_tree))
        assert len(census.ranking) > 1
        assert oc.render_markdown(census, top=1) in out.read_text()

    def test_two_runs_write_identical_bytes(self, pytest_tree, tmp_path):
        out = tmp_path / 'report.md'
        argv = _argv('pytest', pytest_tree, out)
        assert suite_census.main(argv) == 0
        first = out.read_bytes()
        assert suite_census.main(argv) == 0
        assert out.read_bytes() == first


def test_the_nextest_report_carries_the_rust_duplication_table(tmp_path):
    project = nextest_project(tmp_path / 'reify')
    out = tmp_path / 'report.md'
    assert suite_census.main(_argv('nextest', project, out)) == 0
    assert rust.render_markdown(rust.measure_rust_tree(project)) in out.read_text()


class TestRefusals:
    def test_a_missing_root_exits_2_naming_it_and_writes_nothing(self, tmp_path, capsys):
        missing = tmp_path / 'nowhere'
        out = tmp_path / 'report.md'
        argv = _argv('pytest', missing, out)
        assert suite_census.main(argv) == 2
        assert str(missing) in capsys.readouterr().err
        assert not out.exists()

    def test_a_tree_below_the_work_tree_top_exits_2_with_the_metrics_error(
        self, pytest_tree, tmp_path, capsys,
    ):
        out = tmp_path / 'report.md'
        argv = [
            '--ecosystem', 'pytest', '--root', str(pytest_tree),
            '--tree', str(pytest_tree / 'orchestrator'), '--project', 'demo', '--out', str(out),
        ]
        assert suite_census.main(argv) == 2
        assert 'not the top of a git work tree' in capsys.readouterr().err
        assert not out.exists()

    def test_an_unknown_ecosystem_is_an_argparse_error(self, tmp_path):
        with pytest.raises(SystemExit) as exit_info:
            suite_census.main(_argv('maven', tmp_path, tmp_path / 'report.md'))
        assert exit_info.value.code == 2
