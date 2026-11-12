"""The Rust census finds #[test] fns with a comment- and string-aware lexer and counts duplication per crate."""
from __future__ import annotations

import pytest
import suite_census_rust as rust
from suite_census_fixtures import git_tree


def _names(source: str) -> list[str]:
    return [fn.name for fn in rust.rust_test_fns(source)]


class TestLexer:
    def test_a_brace_inside_a_string_does_not_end_the_body(self):
        (fn,) = rust.rust_test_fns('#[test] fn a() { let s = "}"; assert!(true); }')
        assert fn.name == 'a'
        assert 'assert!(true);' in fn.body_lines[0]

    def test_a_tokio_test_with_arguments_is_found(self):
        assert _names('#[tokio::test(flavor = "multi_thread")] async fn b() {}') == ['b']

    def test_other_attributes_and_qualifiers_are_skipped(self):
        (fn,) = rust.rust_test_fns('#[test]\n#[should_panic]\npub fn c() {}')
        assert (fn.name, fn.start_line) == ('c', 3)

    def test_cfg_test_is_not_a_test_attribute(self):
        assert _names('#[cfg(test)] mod tests { #[test] fn inner() {} }') == ['inner']

    def test_literals_lifetimes_and_comments_keep_the_body_span(self):
        source = '\n'.join([
            '#[test]',
            'fn tricky() {',
            '    let raw = r#"}"#;',
            '    let bytes = br"{";',
            "    let open = '{';",
            "    let quote = '\\'';",
            "    fn helper<'a>(x: &'a str) -> &'a str { x }",
            '    // }',
            '    /* /* } */ */',
            '    assert!(true);',
            '}',
            '#[test]',
            'fn after() {}',
        ])
        tricky, after = rust.rust_test_fns(source)
        assert (tricky.name, after.name) == ('tricky', 'after')
        stripped = [line.strip() for line in tricky.body_lines]
        assert 'let raw = r#"}"#;' in stripped and 'assert!(true);' in stripped
        assert not [line for line in stripped if '//' in line or '/*' in line or '*/' in line]

    def test_a_one_liner_has_one_body_line(self):
        (fn,) = rust.rust_test_fns('#[test] fn one_liner() { assert_eq!(1, 1); }')
        assert len(fn.body_lines) == 1

    def test_a_plain_fn_is_not_a_test(self):
        assert _names('fn helper() {}\n#[test]\nfn real() { helper(); }') == ['real']


T_RS = '''\
use alpha::parse;

#[test]
fn parse_int_ok() {
    assert!(parse("1") > 0);
}

#[test]
fn parse_int_err() {
    assert!(parse("x") < 0);
}

#[test]
fn parse_int_overflow() {
    if big() {
        overflow();
    }
    run(|| {
        go();
    });

}

#[test]
fn first_check() {
    let v = parse("1");
    assert_eq!(v, 1);
}

#[test]
fn second_check() {
    let v = parse("1");
    check(v);
}
'''

LIB_RS = '''\
pub fn parse(s: &str) -> i32 { 1 }

#[cfg(test)]
mod tests {
    #[test]
    fn lone() {
        assert_eq!(super::parse("1"), 1);
    }
}
'''

U_RS = '''\
#[test]
fn beta_case() {
    let v = parse("1");
}
'''


@pytest.fixture(scope='module')
def census(tmp_path_factory: pytest.TempPathFactory) -> rust.RustDuplicationCensus:
    root = git_tree(tmp_path_factory.mktemp('cargo'), {
        'Cargo.toml': '[workspace]\nmembers = ["crates/*"]\n',
        'crates/alpha/Cargo.toml': '[package]\nname = "alpha"\n',
        'crates/alpha/tests/t.rs': T_RS,
        'crates/alpha/src/lib.rs': LIB_RS,
        'crates/beta/Cargo.toml': '[package]\nname = "beta"\n',
        'crates/beta/tests/u.rs': U_RS,
    })
    (root / 'crates' / 'alpha' / 'tests' / 'extra.rs').write_text('#[test]\nfn extra() {}\n')
    return rust.measure_rust_tree(root)


def _row(census: rust.RustDuplicationCensus, crate: str) -> rust.RustCrateRow:
    (row,) = [row for row in census.rows if row.crate == crate]
    return row


class TestCrateRows:
    def test_rows_are_crates_in_name_order(self, census):
        assert [row.crate for row in census.rows] == ['alpha', 'beta']

    def test_tracked_test_fns_only(self, census):
        alpha = _row(census, 'alpha')
        assert (alpha.test_files, alpha.test_fns) == (2, 6)

    def test_duplicate_lines_within_a_crate(self, census):
        alpha = _row(census, 'alpha')
        assert (alpha.non_trivial_lines, alpha.duplicated_lines, alpha.redundant_lines) == (11, 2, 1)
        assert alpha.duplicated_share == pytest.approx(2 / 11)

    def test_duplicates_are_not_counted_across_crates(self, census):
        beta = _row(census, 'beta')
        assert (beta.non_trivial_lines, beta.duplicated_lines) == (1, 0)

    def test_name_families(self, census):
        alpha = _row(census, 'alpha')
        assert alpha.family_members == 3
        assert alpha.largest_family == (3, 'crates/alpha/tests/t.rs', 'parse_int')
        assert alpha.family_share == pytest.approx(3 / 6)

    def test_totals_sum_the_crates(self, census):
        totals = census.totals
        assert (totals.test_fns, totals.non_trivial_lines, totals.duplicated_lines) == (7, 12, 2)
        assert (totals.redundant_lines, totals.family_members) == (1, 3)
        assert census.complete is True


def test_render_has_one_row_per_crate_plus_totals(census):
    text = rust.render_markdown(census)
    first_cells = [
        line.strip('|').split('|')[0].strip() for line in text.splitlines() if line.startswith('|')
    ]
    assert first_cells.count('alpha') == 1 and first_cells.count('beta') == 1
    assert first_cells.index('alpha') < first_cells.index('beta') < first_cells.index(
        census.totals.crate
    )
