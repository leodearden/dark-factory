"""Tests of ``scripts/rust_metrics_snapshot.py``, the Cargo-workspace metrics report.

Per-project rules are a TOML file loaded at the boundary into a typed value.
tokei and cargo are shims first on PATH; lizard is the real, uv-locked one.
The fixture rules and the shims are in rust_snapshot_fixtures.py.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import rust_metrics_snapshot
import source_measures
from git_listing import git
from quality_snapshot_fixtures import BASE_FILES, committed_repo, run_snapshot
from quality_snapshot_identity import below_run_id_and_cost
from rust_snapshot_fixtures import (
    CARGO_PRINTS_ENV,
    FIXTURE_RULES_TOML,
    MOVES_HEAD_ENV,
    RUST_CYCLE_FILES,
    RUST_LAYERING_FILES,
    RUST_WORKSPACE_FILES,
    TOKEI_PRINTS_ENV,
    rust_workspace_repo,
    tool_shims,
    write_fixture_rules,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]
REIFY_RULES = REPO_ROOT / 'scripts' / 'rust_metrics_rules' / 'reify.toml'
INSTRUMENT_SCRIPT = REPO_ROOT / 'scripts' / 'rust_metrics_snapshot.py'


class TestReifyRules:
    @pytest.fixture
    def rules(self) -> rust_metrics_snapshot.RustRules:
        return rust_metrics_snapshot.load_rules(REIFY_RULES)

    def test_ordered_area_rules_take_the_first_match(self, rules: rust_metrics_snapshot.RustRules) -> None:
        assert rules.area_of('crates/reify-eval/src/compute_targets/x.rs') == 'analysis'
        assert rules.area_of('crates/reify-eval/src/lib.rs') == 'language-pipeline'

    def test_a_path_no_rule_maps_has_no_area(self, rules: rust_metrics_snapshot.RustRules) -> None:
        assert rules.area_of('nowhere/x.rs') is None

    def test_the_exclude_pattern_matches_from_the_start(self, rules: rust_metrics_snapshot.RustRules) -> None:
        assert rules.excluded('docs/a.rs')
        assert not rules.excluded('crates/reify-core/src/lib.rs')

    def test_the_test_pattern_searches_the_path(self, rules: rust_metrics_snapshot.RustRules) -> None:
        assert rules.is_test('crates/reify-core/tests/a.rs')
        assert not rules.is_test('crates/reify-core/src/lib.rs')


def test_to_json_gives_sources_in_order_and_an_iso_date(tmp_path: Path) -> None:
    rules = rust_metrics_snapshot.load_rules(write_fixture_rules(tmp_path / 'fixture.toml'))
    as_json = rules.to_json()
    assert json.loads(json.dumps(as_json)) == as_json
    assert as_json == {
        'exclude': '^docs/',
        'test': '(^|/)tests?/',
        'area': [
            {'pattern': '^crates/core/', 'area': 'core'},
            {'pattern': '^crates/app/', 'area': 'app'},
        ],
        'inferred': {},
        'lizard_excludes': ['./target/*'],
        'fix_density_since': '2026-07-06',
    }


def _replacing(old: str, new: str) -> Callable[[str], str]:
    def edit(text: str) -> str:
        assert text.count(old) == 1, old
        return text.replace(old, new)

    return edit


@pytest.mark.parametrize(
    ('edit', 'key'),
    [
        pytest.param(_replacing("test = '(^|/)tests?/'\n", ''), 'test', id='missing-test'),
        pytest.param(_replacing('fix_density_since = 2026-07-06\n', ''), 'fix_density_since', id='missing-date'),
        pytest.param(
            _replacing('fix_density_since = 2026-07-06', "fix_density_since = 'soon'"),
            'fix_density_since',
            id='date-not-a-date',
        ),
        pytest.param(_replacing("exclude = '^docs/'", "exclude = '^(docs/'"), 'exclude', id='exclude-regex'),
        pytest.param(
            _replacing("pattern = '^crates/app/'", "pattern = '^crates/(app/'"),
            'area[1].pattern',
            id='area-regex',
        ),
        pytest.param(_replacing("pattern = '^crates/app/'\n", ''), 'area[1].pattern', id='area-without-pattern'),
        pytest.param(_replacing("area = 'app'\n", ''), 'area[1].area', id='area-without-area'),
    ],
)
def test_a_malformed_rules_file_is_refused_naming_the_file_and_the_key(
    tmp_path: Path, edit: Callable[[str], str], key: str
) -> None:
    path = tmp_path / 'bad.toml'
    path.write_text(edit(FIXTURE_RULES_TOML), encoding='utf-8')
    with pytest.raises(source_measures.MetricsError) as refused:
        rust_metrics_snapshot.load_rules(path)
    assert str(path) in str(refused.value)
    assert key in str(refused.value)


@pytest.fixture
def fixture_rules(tmp_path: Path) -> Path:
    return write_fixture_rules(tmp_path / 'rules' / 'fixture.toml')


def _put_first_on_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')


class TestParams:
    @pytest.mark.parametrize('tokei_version', ['15.0.0', '99.1.2'])
    def test_params_records_the_versions_of_the_tools_that_measure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fixture_rules: Path,
        tokei_version: str,
    ) -> None:
        _put_first_on_path(monkeypatch, tool_shims(tmp_path / 'bin', tokei_version=tokei_version))
        assert rust_metrics_snapshot.main(['--params', '--rules', str(fixture_rules)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert len(printed) == 1
        rules = rust_metrics_snapshot.load_rules(fixture_rules)
        assert json.loads(printed[0]) == rust_metrics_snapshot.measurement_params(rules)
        assert json.loads(printed[0]) == {
            'tokei_version': tokei_version,
            'lizard_version': importlib.metadata.version('lizard'),
            'python_version': platform.python_version(),
            'instrument': {
                'rust_metrics_snapshot': git(REPO_ROOT, 'hash-object', 'scripts/rust_metrics_snapshot.py').strip(),
            },
            'h14_soft_ceiling_lines': 1500,
            'h14_alarm_lines': 2000,
            'rules': rules.to_json(),
        }

    def test_a_missing_tokei_is_refused_naming_its_install_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fixture_rules: Path,
    ) -> None:
        bin_dir = tool_shims(tmp_path / 'bin')
        (bin_dir / 'tokei').unlink()
        real_git = shutil.which('git')
        assert real_git is not None
        git_only = tmp_path / 'git-only'
        git_only.mkdir()
        (git_only / 'git').symlink_to(real_git)
        monkeypatch.setenv('PATH', os.pathsep.join([str(bin_dir), str(git_only)]))
        assert rust_metrics_snapshot.main(['--params', '--rules', str(fixture_rules)]) == 2
        refusal = capsys.readouterr().err
        assert 'tokei' in refusal
        assert 'cargo install --locked tokei' in refusal


@pytest.fixture
def shims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tool_shims(tmp_path / 'bin')
    _put_first_on_path(monkeypatch, bin_dir)
    return bin_dir


def _measure(rules: Path, root: Path, out: Path, run_id: str = 'r1') -> int:
    return rust_metrics_snapshot.main(
        ['--rules', str(rules), '--run-id', run_id, '--out', str(out), '--root', str(root)]
    )


@pytest.mark.usefixtures('shims')
class TestTheMeasurement:
    def test_it_writes_one_snapshot_whose_params_are_the_predicted_ones(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], fixture_rules: Path
    ) -> None:
        repo = rust_workspace_repo(tmp_path)
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, repo, out) == 0
        assert list(out.parent.iterdir()) == [out]
        assert git(repo, 'status', '--porcelain') == ''
        taken = rust_metrics_snapshot.load_snapshot(out)
        assert taken['instrument'] == rust_metrics_snapshot.INSTRUMENT == 'rust-metrics-snapshot'
        assert taken['schema_version'] == rust_metrics_snapshot.SCHEMA_VERSION
        assert taken['as_of_sha'] == git(repo, 'rev-parse', 'HEAD').strip()
        assert taken['params'] == rust_metrics_snapshot.measurement_params(
            rust_metrics_snapshot.load_rules(fixture_rules)
        )
        assert {record['path']: (record['area'], record['kind']) for record in taken['files']} == {
            'crates/app/src/main.rs': ('app', 'src'),
            'crates/core/src/lib.rs': ('core', 'src'),
            'crates/core/tests/t.rs': ('core', 'tests'),
        }
        assert 'metrics snapshot r1 @ ' in capsys.readouterr().out

    def test_it_measures_complexity_internal_items_imports_and_area_lines(
        self, tmp_path: Path, fixture_rules: Path
    ) -> None:
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, rust_workspace_repo(tmp_path), out) == 0
        taken = rust_metrics_snapshot.load_snapshot(out)
        files = {record['path']: record for record in taken['files']}
        lib, main = files['crates/core/src/lib.rs'], files['crates/app/src/main.rs']
        assert (lib['ccn_max'], lib['ccn_max_fn'], lib['internal_pub_items']) == (3, 'classify', 1)
        assert (lib['fan_in_src'], main['fan_out']) == (1, 1)
        assert taken['import_graph']['edges'] == [['app', 'core']]
        tests_lines = RUST_WORKSPACE_FILES['crates/core/tests/t.rs'].count('\n')
        assert taken['per_area']['core']['tests_lines'] == tests_lines

    def test_it_counts_reach_back_deferred_and_cycle_closing_imports(
        self, tmp_path: Path, fixture_rules: Path
    ) -> None:
        out = tmp_path / 'out' / 'snapshot.json'
        repo = rust_workspace_repo(tmp_path, {**RUST_WORKSPACE_FILES, **RUST_LAYERING_FILES})
        assert _measure(fixture_rules, repo, out) == 0
        taken = rust_metrics_snapshot.load_snapshot(out)
        counted = ('reach_back_imports', 'deferred_imports', 'cycle_closing_imports')
        files = {record['path']: record for record in taken['files']}
        assert {path: tuple(files[path][key] for key in counted) for path in RUST_LAYERING_FILES} == {
            'crates/core/src/lib.rs': (0, 0, 0),
            'crates/core/src/a.rs': (1, 0, 0),
            'crates/core/src/b.rs': (0, 1, 1),
        }
        assert tuple(taken['per_area']['core'][key] for key in counted) == (1, 1, 1)
        graph = taken['import_graph']
        assert [(entry['from'], entry['to']) for entry in graph['reach_back']] == [('core::a', 'core')]
        assert [(entry['from'], entry['to'], entry['closes_cycle']) for entry in graph['deferred']] == [
            ('core::b', 'core::a', True)
        ]
        assert graph['cycles'] == [['core', 'core::a']]
        assert graph['hidden_cycles'] == [['core', 'core::a', 'core::b']]

    def test_an_unreadable_domain_file_is_named_and_the_evidence_is_incomplete(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], fixture_rules: Path
    ) -> None:
        repo = rust_workspace_repo(tmp_path)
        gone = 'crates/core/src/gone.rs'
        (repo / gone).symlink_to('missing.rs')
        git(repo, 'add', gone)
        git(repo, '-c', 'user.name=f', '-c', 'user.email=f@e.invalid', 'commit', '--no-verify', '-q', '-m', 'gone')
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, repo, out) == 0
        taken = rust_metrics_snapshot.load_snapshot(out)
        evidence = taken['evidence']
        assert (evidence['complete'], evidence['unreadable'], evidence['domain_files']) == (False, [gone], 4)
        assert gone not in {record['path'] for record in taken['files']}
        printed = capsys.readouterr()
        assert f'unreadable: {gone}: ' in printed.err
        assert f'  {gone}' in printed.out.splitlines()

    def test_two_trees_of_one_commit_measure_identically_below_run_id_and_cost(
        self, tmp_path: Path, fixture_rules: Path
    ) -> None:
        repo = rust_workspace_repo(tmp_path)
        trees = [tmp_path / 'tree-a', tmp_path / 'tree-b']
        outs = [tmp_path / 'out' / 'a.json', tmp_path / 'out' / 'b.json']
        for number, (tree, out) in enumerate(zip(trees, outs, strict=True)):
            git(repo, 'worktree', 'add', '--detach', '-q', str(tree), 'main')
            assert _measure(fixture_rules, tree, out, run_id=f'r{number}') == 0
        assert below_run_id_and_cost(outs[0]) == below_run_id_and_cost(outs[1])
        for out in outs:
            text = out.read_text(encoding='utf-8')
            assert not [tree for tree in trees if str(tree) in text]

    def test_processes_with_different_hash_seeds_measure_identically(
        self, tmp_path: Path, fixture_rules: Path
    ) -> None:
        repo = rust_workspace_repo(tmp_path, {**RUST_WORKSPACE_FILES, **RUST_CYCLE_FILES})
        outs = []
        for seed in range(4):
            out = tmp_path / 'out' / f'{seed}.json'
            argv = ['--rules', str(fixture_rules), '--run-id', f'r{seed}', '--out', str(out), '--root', str(repo)]
            completed = subprocess.run(
                [sys.executable, str(INSTRUMENT_SCRIPT), *argv],
                env={**os.environ, 'PYTHONHASHSEED': str(seed)},
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            outs.append(out)
        assert len(json.loads(outs[0].read_text(encoding='utf-8'))['import_graph']['cycles']) == 2
        assert [below_run_id_and_cost(out) for out in outs[1:]] == [below_run_id_and_cost(outs[0])] * 3


@pytest.mark.usefixtures('shims')
class TestTheRefusals:
    def test_a_head_moved_during_the_measurement_is_refused_and_nothing_is_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fixture_rules: Path,
    ) -> None:
        repo = rust_workspace_repo(tmp_path)
        started_at = git(repo, 'rev-parse', 'HEAD').strip()
        monkeypatch.setenv(MOVES_HEAD_ENV, '1')
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, repo, out) == 2
        found_at = git(repo, 'rev-parse', 'HEAD').strip()
        assert found_at != started_at
        refusal = capsys.readouterr().err
        assert started_at in refusal
        assert found_at in refusal
        assert not out.exists()

    @pytest.mark.parametrize(
        ('env', 'printed', 'named'),
        [
            pytest.param(TOKEI_PRINTS_ENV, 'not json', ('`tokei -f -o json .`', 'no JSON'), id='tokei-not-json'),
            pytest.param(
                TOKEI_PRINTS_ENV,
                '{"Rust": {"reports": [{"name": "./x.rs"}]}}',
                ('`tokei -f -o json .`', "'stats'"),
                id='tokei-report-without-stats',
            ),
            pytest.param(
                CARGO_PRINTS_ENV,
                '{"crates": []}',
                ('`cargo metadata --no-deps --format-version 1`', "'packages'"),
                id='cargo-without-packages',
            ),
        ],
    )
    def test_tool_output_of_another_shape_is_refused_naming_the_command_and_what_it_lacked(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fixture_rules: Path,
        env: str,
        printed: str,
        named: tuple[str, str],
    ) -> None:
        repo = rust_workspace_repo(tmp_path)
        monkeypatch.setenv(env, printed)
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, repo, out) == 2
        refusal = capsys.readouterr().err
        assert [fragment for fragment in named if fragment not in refusal] == []
        assert not out.exists()

    def test_a_dirty_tracked_file_is_refused_by_name_and_nothing_is_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], fixture_rules: Path
    ) -> None:
        repo = rust_workspace_repo(tmp_path)
        (repo / 'crates/core/src/lib.rs').write_text('pub fn changed() {}\n', encoding='utf-8')
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, repo, out) == 2
        assert 'crates/core/src/lib.rs' in capsys.readouterr().err
        assert not out.exists()


class TestLoadSnapshot:
    @pytest.fixture
    def taken(self, tmp_path: Path, fixture_rules: Path, shims: Path) -> dict[str, Any]:
        out = tmp_path / 'out' / 'snapshot.json'
        assert _measure(fixture_rules, rust_workspace_repo(tmp_path), out) == 0
        return json.loads(out.read_text(encoding='utf-8'))

    def _refusal(self, path: Path) -> str:
        with pytest.raises(source_measures.MetricsError) as refused:
            rust_metrics_snapshot.load_snapshot(path)
        assert str(path) in str(refused.value)
        return str(refused.value)

    def test_a_python_instrument_snapshot_is_refused_naming_the_instrument(self, tmp_path: Path) -> None:
        python_snapshot = tmp_path / 'python.json'
        (tmp_path / 'python').mkdir()
        assert run_snapshot(committed_repo(tmp_path / 'python', BASE_FILES), python_snapshot) == 0
        assert 'instrument' in self._refusal(python_snapshot)

    def test_another_schema_version_is_refused_as_another_schema(
        self, tmp_path: Path, taken: dict[str, Any]
    ) -> None:
        taken['schema_version'] = rust_metrics_snapshot.SCHEMA_VERSION + 1
        edited = tmp_path / 'edited.json'
        edited.write_text(json.dumps(taken), encoding='utf-8')
        with pytest.raises(rust_metrics_snapshot.OtherSchemaError) as refused:
            rust_metrics_snapshot.load_snapshot(edited)
        assert str(edited) in str(refused.value)
        assert 'schema_version' in str(refused.value)

    def test_params_missing_a_recorded_key_is_refused_naming_the_key(
        self, tmp_path: Path, taken: dict[str, Any]
    ) -> None:
        del taken['params']['tokei_version']
        edited = tmp_path / 'edited.json'
        edited.write_text(json.dumps(taken), encoding='utf-8')
        assert 'tokei_version' in self._refusal(edited)

    def test_a_file_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        garbled = tmp_path / 'garbled.json'
        garbled.write_text('{"instrument": ', encoding='utf-8')
        assert 'JSON' in self._refusal(garbled)
