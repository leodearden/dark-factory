from orchestrator.config import PriceEntry
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.metrics import RoleUnderTest
from orchestrator.evals.prices import (
    build_pairwise_price_table,
    build_price_table,
    format_price_table_section,
)

# ---------------------------------------------------------------------------
# Task 2477 step-07: build_price_table — per-config C4 price table
# ---------------------------------------------------------------------------

def _cfg(name, model, role='implementer'):
    return EvalConfig(
        name=name, backend='claude', model=model, effort='high', role=RoleUnderTest(role),
    )


class TestBuildPriceTable:
    """{config_name: {role: {input_per_1m, output_per_1m}}} pulled from prices
    by model — an unlisted model yields an EXPLICIT unpriced marker, never a
    silently fabricated price.
    """

    def test_priced_configs_map_role_to_price_entry(self):
        prices = {
            'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0},
            'model-y': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
        }
        configs = [
            _cfg('cfg-a', 'model-x', role='implementer'),
            _cfg('cfg-b', 'model-y', role='architect'),
        ]
        table = build_price_table(configs, prices)
        assert table['cfg-a'] == {
            'implementer': {'input_per_1m': 2.0, 'output_per_1m': 8.0},
        }
        assert table['cfg-b'] == {
            'architect': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
        }

    def test_unlisted_model_gets_explicit_unpriced_marker(self):
        prices = {'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0}}
        configs = [_cfg('cfg-z', 'unlisted-model', role='implementer')]
        table = build_price_table(configs, prices)
        entry = table['cfg-z']['implementer']
        # An EXPLICIT marker, NOT a fabricated price.
        assert 'input_per_1m' not in entry
        assert entry.get('source') == 'unpriced'

    def test_accepts_price_entry_objects(self):
        prices = {'model-x': PriceEntry(input_per_1m=2.0, output_per_1m=8.0)}
        table = build_price_table([_cfg('cfg-a', 'model-x')], prices)
        assert table['cfg-a']['implementer'] == {
            'input_per_1m': 2.0, 'output_per_1m': 8.0,
        }

    def test_deterministic_sorted_config_keys(self):
        prices = {'m': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}
        configs = [_cfg('z', 'm'), _cfg('a', 'm'), _cfg('m', 'm')]
        table = build_price_table(configs, prices)
        assert list(table.keys()) == ['a', 'm', 'z']


# ---------------------------------------------------------------------------
# Task 2478 μ (amend): build_pairwise_price_table — the combined arch+impl
# price table for the end-to-end (matrix/confirm) stages. Keyed to match
# run_end_to_end's ``f'{arch.name}+{impl.name}'`` config_name so the rendered
# price section aligns with the end-to-end composite rows (was individual-keyed).
# ---------------------------------------------------------------------------

class TestBuildPairwisePriceTable:
    """{arch+impl: {architect: entry, implementer: entry}} — one combined-name
    entry per (architect, implementer) pair, carrying BOTH roles' rates.
    """

    def test_pair_keys_combined_name_with_both_roles(self):
        prices = {
            'arch-model': {'input_per_1m': 3.0, 'output_per_1m': 12.0},
            'impl-model': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
        }
        arch = _cfg('architect-x', 'arch-model', role='architect')
        impl = _cfg('impl-y', 'impl-model', role='implementer')
        table = build_pairwise_price_table([(arch, impl)], prices)
        # Key is the combined name run_end_to_end stamps; both roles present.
        assert table == {
            'architect-x+impl-y': {
                'architect': {'input_per_1m': 3.0, 'output_per_1m': 12.0},
                'implementer': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
            }
        }

    def test_unlisted_model_gets_explicit_unpriced_marker(self):
        arch = _cfg('a', 'unlisted', role='architect')
        impl = _cfg('b', 'unlisted', role='implementer')
        table = build_pairwise_price_table([(arch, impl)], {})
        # EXPLICIT marker for each role, never a fabricated price.
        assert table['a+b']['architect'] == {'source': 'unpriced'}
        assert table['a+b']['implementer'] == {'source': 'unpriced'}

    def test_deterministic_sorted_combined_keys(self):
        prices = {'m': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}
        i1 = _cfg('impl-b', 'm', role='implementer')
        pairs = [
            (_cfg('arch-z', 'm', role='architect'), i1),
            (_cfg('arch-a', 'm', role='architect'), i1),
        ]
        table = build_pairwise_price_table(pairs, prices)
        assert list(table.keys()) == ['arch-a+impl-b', 'arch-z+impl-b']


class TestFormatPriceTableSection:
    """The renderer reads exactly the shape the builders write."""

    _PRICES = {'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0}}

    @staticmethod
    def _rows(lines):
        return [line.split() for line in lines[3:]]

    def test_block_opens_with_the_price_table_header(self):
        lines = format_price_table_section(
            build_price_table([_cfg('cfg-a', 'model-x')], self._PRICES),
        )
        assert lines[0] == 'price table:'
        assert lines[1].split() == ['config', 'role', 'input_per_1m', 'output_per_1m']

    def test_listed_entry_renders_its_rates_at_four_dp(self):
        lines = format_price_table_section(
            build_price_table([_cfg('cfg-a', 'model-x')], self._PRICES),
        )
        assert self._rows(lines) == [['cfg-a', 'implementer', '2.0000', '8.0000']]

    def test_unpriced_entry_renders_the_marker_in_both_rate_cells(self):
        lines = format_price_table_section(
            build_price_table([_cfg('cfg-z', 'unlisted-model')], self._PRICES),
        )
        assert self._rows(lines) == [['cfg-z', 'implementer', 'unpriced', 'unpriced']]

    def test_rows_sort_by_config(self):
        configs = [_cfg('z', 'model-x'), _cfg('a', 'unlisted-model')]
        lines = format_price_table_section(build_price_table(configs, self._PRICES))
        assert [row[0] for row in self._rows(lines)] == ['a', 'z']

    def test_rows_sort_by_role_within_a_config(self):
        # The pair puts the implementer first, so the builder inserts it first;
        # the renderer still lists the architect row first.
        pair = (
            _cfg('p', 'model-x', role='implementer'),
            _cfg('q', 'model-x', role='architect'),
        )
        lines = format_price_table_section(
            build_pairwise_price_table([pair], self._PRICES),
        )
        assert [row[:2] for row in self._rows(lines)] == [
            ['p+q', 'architect'], ['p+q', 'implementer'],
        ]
