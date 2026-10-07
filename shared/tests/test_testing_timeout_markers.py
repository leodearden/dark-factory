"""Contract of shared/src/shared/testing_timeout_markers.py, over inline fixtures."""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Mapping
from types import MappingProxyType

import pytest
from shared.testing_timeout_markers import (
    DELIBERATE_TIGHT_BOUND_CEILING,
    SiteKind,
    TimeoutSite,
    inverts,
    timeout_marker_sites,
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
