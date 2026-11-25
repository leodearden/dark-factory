"""Tests of ``scripts/rust_metrics_snapshot.py``, the Cargo-workspace metrics report.

Per-project rules are a TOML file loaded at the boundary into a typed value;
the fixture rules are in rust_snapshot_fixtures.py.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import rust_metrics_snapshot
import source_measures
from rust_snapshot_fixtures import FIXTURE_RULES_TOML, write_fixture_rules

REPO_ROOT = Path(__file__).parents[2]
REIFY_RULES = REPO_ROOT / 'scripts' / 'rust_metrics_rules' / 'reify.toml'


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
