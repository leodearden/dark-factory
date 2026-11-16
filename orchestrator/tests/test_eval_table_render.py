from orchestrator.evals.table_render import (
    ci_cell,
    optional_float_cell,
    render_fixed_table,
)


class TestRenderFixedTable:
    """The one width-computed fixed-table renderer every deterministic report table shares."""

    _COLUMNS = ('name', 'score')

    def test_widths_are_max_of_header_and_cells(self):
        lines = render_fixed_table(
            self._COLUMNS,
            [{'name': 'a-very-long-config', 'score': '1.0000'}],
            header='demo:',
        )
        assert lines[0] == 'demo:'
        # The header cell is padded out to the widest row cell, so the columns
        # line up; the dashes rule matches those same widths.
        assert lines[1].startswith('name'.ljust(len('a-very-long-config')))
        assert len(lines[2].split('  ')[0]) == len('a-very-long-config')
        assert lines[3].startswith('a-very-long-config')

    def test_trailing_padding_is_stripped(self):
        # Invisible trailing whitespace makes an otherwise-identical table
        # differ byte-for-byte, and it hides a line's real ending — the
        # per-config mean block asserts on a trailing '-' meaning "nothing
        # scored", which must never read as 0.0000.
        lines = render_fixed_table(
            self._COLUMNS,
            [{'name': 'a', 'score': 'a-long-score-cell'}, {'name': 'b', 'score': '-'}],
            header='demo:',
        )
        assert all(ln == ln.rstrip() for ln in lines)
        assert lines[-1].endswith('-')

    def test_empty_rows_still_render_headers_and_rule(self):
        # No rows must not mean no table: the header/rule still render at the
        # header widths rather than raising on the max() of an empty sequence.
        lines = render_fixed_table(self._COLUMNS, [], header='demo:')
        assert lines[0] == 'demo:'
        assert lines[1] == 'name  score'
        assert lines[2] == '----  -----'
        assert len(lines) == 3

    def test_header_is_optional(self):
        lines = render_fixed_table(self._COLUMNS, [])
        assert lines[0] == 'name  score'


class TestOptionalFloatCell:
    """An unmeasured value renders '-', never a score of zero."""

    def test_none_renders_dash(self):
        assert optional_float_cell(None) == '-'

    def test_value_renders_at_four_dp_by_default(self):
        assert optional_float_cell(0.5) == '0.5000'

    def test_precision_overrides_the_default(self):
        assert optional_float_cell(1.23456, precision=2) == '1.23'


class TestCiCell:
    """A mean_ci95-shaped dict renders as its [lo, hi] interval; no interval renders '-'."""

    def test_none_renders_dash(self):
        assert ci_cell(None) == '-'

    def test_empty_dict_renders_dash(self):
        assert ci_cell({}) == '-'

    def test_interval_renders_lo_and_hi_at_four_dp(self):
        ci = {'mean': 0.175, 'lo': 0.1, 'hi': 0.25, 'n': 3, 'sufficient': True}
        assert ci_cell(ci) == '[0.1000, 0.2500]'
