"""Contract of shared/src/shared/testing_timeout_markers.py."""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from shared.testing_timeout_markers import (
    DELIBERATE_TIGHT_BOUND_CEILING,
    SiteKind,
    TimeoutSite,
    inverts,
    scan_python_tree,
    timeout_marker_sites,
    verify_cli_timeout,
)

_NO_SANCTIONED_NAMES: Mapping[str, float] = MappingProxyType({})


def _parsed(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _sites(
    source: str, sanctioned: Mapping[str, float] = _NO_SANCTIONED_NAMES
) -> dict[str, float | None]:
    return {site.qualname: site.seconds for site in timeout_marker_sites(_parsed(source), sanctioned)}


# ---------------------------------------------------------------------------
# timeout_marker_sites -- the extractor.
# ---------------------------------------------------------------------------


def test_extractor_reads_a_bare_literal_function_decorator() -> None:
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(120)
        def test_slow() -> None:
            pass
        """
    ) == {'test_slow': 120.0}


def test_extractor_reads_the_keyword_spelling() -> None:
    """``timeout(timeout=120)`` clamps exactly as hard as the positional form."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(timeout=120)
        def test_slow() -> None:
            pass
        """
    ) == {'test_slow': 120.0}


def test_extractor_reads_module_level_pytestmark() -> None:
    assert _sites(
        """
        import pytest

        pytestmark = pytest.mark.timeout(120)
        """
    ) == {'<module>': 120.0}


def test_extractor_reads_list_form_pytestmark() -> None:
    assert _sites(
        """
        import pytest

        pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]
        """
    ) == {'<module>': 120.0}


def test_extractor_reads_a_class_decorator() -> None:
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(180)
        class TestThing:
            def test_a(self) -> None:
                pass
        """
    ) == {'TestThing': 180.0}


def test_extractor_reads_class_level_pytestmark() -> None:
    assert _sites(
        """
        import pytest

        class TestThing:
            pytestmark = pytest.mark.timeout(180)

            def test_a(self) -> None:
                pass
        """
    ) == {'TestThing::<pytestmark>': 180.0}


def test_extractor_gives_a_method_a_dotted_qualname() -> None:
    """Same-named methods in two classes must not collapse into one key."""
    assert _sites(
        """
        import pytest

        class TestOne:
            @pytest.mark.timeout(120)
            def test_a(self) -> None:
                pass

        class TestTwo:
            @pytest.mark.timeout(150)
            def test_a(self) -> None:
                pass
        """
    ) == {'TestOne::test_a': 120.0, 'TestTwo::test_a': 150.0}


_NAMED_MARKS_SOURCE = """
    import pytest

    @pytest.mark.timeout(BIG)
    def test_bare() -> None:
        pass

    @pytest.mark.timeout(helpers.BIG)
    def test_dotted() -> None:
        pass

    @pytest.mark.timeout(ABSENT)
    def test_absent() -> None:
        pass
    """


def test_extractor_resolves_names_through_the_sanctioned_map() -> None:
    """Bare and dotted names resolve on the trailing name; an absent name is None."""
    assert _sites(_NAMED_MARKS_SOURCE, {'BIG': 540.0}) == {
        'test_bare': 540.0,
        'test_dotted': 540.0,
        'test_absent': None,
    }


def test_an_empty_sanctioned_map_resolves_no_name() -> None:
    assert _sites(_NAMED_MARKS_SOURCE) == {
        'test_bare': None,
        'test_dotted': None,
        'test_absent': None,
    }


def test_extractor_yields_none_for_an_unresolvable_expression() -> None:
    """A computed argument is "no opinion", never a guessed number."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(int(ROW_BUDGET * 2))
        def test_derived() -> None:
            pass
        """,
        {'ROW_BUDGET': 100.0},
    ) == {'test_derived': None}


def test_extractor_yields_none_for_a_zero_arg_timeout_mark() -> None:
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout()
        def test_a() -> None:
            pass

        @pytest.mark.timeout(method='signal')
        def test_b() -> None:
            pass
        """
    ) == {'test_a': None, 'test_b': None}


def test_extractor_does_not_read_a_bool_as_seconds() -> None:
    """``bool`` subclasses ``int``; read naively, ``True`` would pin 1s."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(True)
        def test_a() -> None:
            pass
        """
    ) == {'test_a': None}


def test_extractor_ignores_marks_that_are_not_timeout() -> None:
    """A non-``timeout`` mark is not a site at all -- not a site with None."""
    assert _sites(
        """
        import pytest

        @pytest.mark.asyncio
        @pytest.mark.slow
        def test_a() -> None:
            pass
        """
    ) == {}


def test_a_site_records_where_and_how_the_mark_is_spelled() -> None:
    tree = _parsed(
        """
        import pytest

        pytestmark = pytest.mark.timeout(120)

        @pytest.mark.timeout(BIG)
        class TestThing:
            pytestmark = [pytest.mark.timeout()]

            @pytest.mark.timeout(timeout=30)
            def test_a(self) -> None:
                pass
        """
    )

    assert timeout_marker_sites(tree, {'BIG': 540.0}) == (
        TimeoutSite('<module>', SiteKind.MODULE_PYTESTMARK, 120.0, 4, '120'),
        TimeoutSite('TestThing', SiteKind.CLASS_DECORATOR, 540.0, 6, 'BIG'),
        TimeoutSite('TestThing::<pytestmark>', SiteKind.CLASS_PYTESTMARK, None, 8, ''),
        TimeoutSite('TestThing::test_a', SiteKind.DECORATOR, 30.0, 10, '30'),
    )


def test_site_kinds_are_the_closed_vocabulary_failure_messages_render() -> None:
    assert {kind.value for kind in SiteKind} == {
        'decorator',
        'class-decorator',
        'module-pytestmark',
        'class-pytestmark',
    }
    assert f'{SiteKind.CLASS_DECORATOR}' == 'class-decorator'


# ---------------------------------------------------------------------------
# inverts -- the band.
# ---------------------------------------------------------------------------


def test_the_deliberate_tight_bound_ceiling_is_sixty_seconds() -> None:
    assert DELIBERATE_TIGHT_BOUND_CEILING == 60


def test_the_band_edges_are_exactly_where_the_design_puts_them() -> None:
    """True only strictly inside ``(ceiling, budget)``; None has no opinion."""
    assert [
        inverts(seconds, verify_cli_budget=300)
        for seconds in (None, 15, 60, 61, 90, 120, 150, 180, 299, 300, 360, 960)
    ] == [
        False,  # None       -- unresolvable: no opinion, never an offence
        False,  # 15         -- test_verify_clock_stop.py's watchdog marks
        False,  # 60         -- exactly the ceiling: a deliberate tight bound
        True,  # 61          -- first inverting value
        True,  # 90          -- measured, test_merge_queue.py
        True,  # 120         -- measured, the named regression instance
        True,  # 150         -- measured, test_offline_lane_integration.py
        True,  # 180         -- measured, the most common in-band value (34 sites)
        True,  # 299         -- last inverting value
        False,  # 300        -- WHOLE_TREE_SCAN / HEAVY_BARRIER / the CLI budget
        False,  # 360        -- loosens under both
        False,  # 960        -- PYTEST_TIMEOUT (warm-lane bash bucket)
    ]


def test_the_top_edge_is_the_budget_passed_in() -> None:
    assert [
        inverts(seconds, verify_cli_budget=600) for seconds in (60, 61, 300, 599, 600)
    ] == [False, True, True, True, False]


@pytest.mark.parametrize('budget', [DELIBERATE_TIGHT_BOUND_CEILING, 30])
def test_a_budget_that_empties_the_band_is_refused(budget: int) -> None:
    """An empty band would make every sweep built on it pass vacuously."""
    with pytest.raises(ValueError) as refused:
        inverts(120, verify_cli_budget=budget)

    assert str(DELIBERATE_TIGHT_BOUND_CEILING) in str(refused.value)
    assert str(budget) in str(refused.value)


# ---------------------------------------------------------------------------
# verify_cli_timeout -- the budget a package's verify test_command passes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('test_command', 'expected'), [
    ('uv run --directory fused-memory pytest tests/ --tb=short -q --timeout=300', 300),
    ('uv run --directory shared pytest tests/ -q --timeout 300', 300),
    ('uv run --directory shared pytest tests/ -q', None),
    (
        'uv run --directory a pytest tests/ --timeout=300 '
        '&& uv run --directory b pytest tests/ --timeout=600',
        300,
    ),
])
def test_verify_cli_timeout_reads_the_first_timeout_flag(test_command, expected) -> None:
    assert verify_cli_timeout(test_command) == expected


# ---------------------------------------------------------------------------
# scan_python_tree -- one fail-soft pass over a real directory tree.
# ---------------------------------------------------------------------------

_MARKED_SOURCE = """\
import pytest


@pytest.mark.timeout(120)
def test_a() -> None:
    pass
"""


@pytest.fixture
def scanned_root(tmp_path: Path) -> Path:
    (tmp_path / 'test_z.py').write_text(_MARKED_SOURCE, encoding='utf-8')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'test_b.py').write_text(_MARKED_SOURCE, encoding='utf-8')
    (tmp_path / 'sub' / 'not_utf8.py').write_bytes(b'\xff\xfe pytestmark = 1\n')
    (tmp_path / 'a_broken.py').write_text('def test_a(:\n', encoding='utf-8')
    (tmp_path / 'notes.txt').write_text(_MARKED_SOURCE, encoding='utf-8')
    return tmp_path


def _sites_by_module(module: str, tree: ast.Module) -> Iterator[tuple[str, TimeoutSite]]:
    for site in timeout_marker_sites(tree, _NO_SANCTIONED_NAMES):
        yield module, site


def test_scan_hands_extract_each_module_keyed_by_its_relative_posix_path(
    scanned_root: Path,
) -> None:
    scan = scan_python_tree(scanned_root, _sites_by_module)

    assert [(module, site.qualname) for module, site in scan.items] == [
        ('sub/test_b.py', 'test_a'),
        ('test_z.py', 'test_a'),
    ]


def test_scan_visits_modules_in_sorted_path_order(scanned_root: Path) -> None:
    visited: list[str] = []

    def record(module: str, tree: ast.Module) -> Iterator[str]:
        visited.append(module)
        yield module

    scan = scan_python_tree(scanned_root, record)

    assert visited == sorted(visited)
    assert scan.items == tuple(visited)


def test_an_unparseable_module_is_examined_but_never_extracted(scanned_root: Path) -> None:
    visited: list[str] = []

    def record(module: str, tree: ast.Module) -> Iterator[str]:
        visited.append(module)
        return iter(())

    scan = scan_python_tree(scanned_root, record)

    assert 'a_broken.py' not in visited
    assert scan.examined == 3


def test_an_undecodable_module_is_counted_unreadable_not_examined(scanned_root: Path) -> None:
    scan = scan_python_tree(scanned_root, _sites_by_module)

    assert scan.unreadable == ('sub/not_utf8.py',)
    assert scan.examined == 3


def test_a_file_that_is_not_python_is_ignored(scanned_root: Path) -> None:
    scan = scan_python_tree(scanned_root, lambda module, tree: iter((module,)))

    assert 'notes.txt' not in scan.items
    assert scan.items == ('sub/test_b.py', 'test_z.py')


def test_one_pass_can_feed_several_extractions_per_module(scanned_root: Path) -> None:
    """Orchestrator extracts marker sites AND constant bindings from each single parse."""

    def sites_and_names(
        module: str, tree: ast.Module
    ) -> Iterator[tuple[str, int, frozenset[str]]]:
        names = frozenset(
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        )
        yield module, len(timeout_marker_sites(tree, _NO_SANCTIONED_NAMES)), names

    scan = scan_python_tree(scanned_root, sites_and_names)

    assert scan.items == (
        ('sub/test_b.py', 1, frozenset({'test_a'})),
        ('test_z.py', 1, frozenset({'test_a'})),
    )


def test_a_scan_result_is_immutable(scanned_root: Path) -> None:
    scan = scan_python_tree(scanned_root, _sites_by_module)

    assert isinstance(scan.items, tuple)
    assert isinstance(scan.unreadable, tuple)
    with pytest.raises(AttributeError):
        scan.examined = 0  # pyright: ignore[reportAttributeAccessIssue]
