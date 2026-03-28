"""Unit tests for the _get_opening_tag helper in test_helpers.py.

These tests are written first (TDD) and will fail until test_helpers.py is
implemented.
"""

import pytest

from .test_helpers import _get_opening_tag


class TestSingleLineTag:
    """Marker appears inside a simple single-line opening tag."""

    def test_returns_full_opening_tag(self):
        html = '<span data-updated-for="orchestrators" class="text-gray-500">hello</span>'
        result = _get_opening_tag(html, 'data-updated-for="orchestrators"')
        assert result == '<span data-updated-for="orchestrators" class="text-gray-500">'

    def test_marker_at_start_of_attributes(self):
        html = '<div data-section="recon" id="main">content</div>'
        result = _get_opening_tag(html, 'data-section="recon"')
        assert result == '<div data-section="recon" id="main">'

    def test_marker_at_end_of_attributes(self):
        html = '<button type="button" data-testid="journal-badge">3</button>'
        result = _get_opening_tag(html, 'data-testid="journal-badge"')
        assert result == '<button type="button" data-testid="journal-badge">'


class TestMultiLineTag:
    """Marker appears inside a tag that spans multiple lines."""

    def test_multiline_tag_extraction(self):
        html = (
            '<span\n'
            '  class="text-gray-500"\n'
            '  data-updated-for="memory"\n'
            '  aria-hidden="true"\n'
            '>updated</span>'
        )
        result = _get_opening_tag(html, 'data-updated-for="memory"')
        assert 'data-updated-for="memory"' in result
        assert result.startswith('<span')
        assert result.endswith('>')

    def test_multiline_tag_contains_all_attributes(self):
        html = (
            '<button\n'
            '  type="button"\n'
            '  data-testid="journal-badge"\n'
            '  aria-label="Show 3 journal entries"\n'
            '>3</button>'
        )
        result = _get_opening_tag(html, 'data-testid="journal-badge"')
        assert 'aria-label="Show 3 journal entries"' in result
        assert result.startswith('<button')


class TestValueErrorOnMissing:
    """ValueError is raised when the marker is not found inside any tag."""

    def test_marker_completely_absent(self):
        html = '<div class="foo">content</div>'
        with pytest.raises(ValueError):
            _get_opening_tag(html, 'data-testid="missing"')

    def test_empty_html(self):
        with pytest.raises(ValueError):
            _get_opening_tag('', 'data-section="recon"')

    def test_error_message_contains_marker(self):
        html = '<div>no match here</div>'
        marker = 'data-testid="nope"'
        with pytest.raises(ValueError, match='data-testid="nope"'):
            _get_opening_tag(html, marker)


class TestMarkerInTextContent:
    """Marker appears in text/attribute values but NOT inside an opening tag."""

    def test_marker_in_text_node_raises(self):
        # The marker text appears between tags, not inside one
        html = '<p>data-section="recon" is a cool attribute</p>'
        with pytest.raises(ValueError):
            _get_opening_tag(html, 'data-section="recon"')

class TestMultipleTags:
    """When multiple tags exist, the one containing the marker is returned."""

    def test_returns_matching_tag_not_first(self):
        html = (
            '<span data-updated-for="orchestrators">A</span>'
            '<span data-updated-for="memory" class="x">B</span>'
        )
        result = _get_opening_tag(html, 'data-updated-for="memory"')
        assert result == '<span data-updated-for="memory" class="x">'

    def test_returns_first_matching_occurrence(self):
        # When a marker appears in two tags, the first is returned
        html = (
            '<div data-testid="journal-badge" class="first">A</div>'
            '<span data-testid="journal-badge" class="second">B</span>'
        )
        result = _get_opening_tag(html, 'data-testid="journal-badge"')
        assert result == '<div data-testid="journal-badge" class="first">'


class TestRegexEscaping:
    """Special regex characters in the marker are escaped properly."""

    def test_quotes_in_marker_are_safe(self):
        # Quotes are the most common special chars in attribute markers
        html = '<button data-testid="journal-badge" type="button">X</button>'
        result = _get_opening_tag(html, 'data-testid="journal-badge"')
        assert 'data-testid="journal-badge"' in result

    def test_marker_with_parentheses(self):
        # Parentheses are regex-special but should be treated literally
        html = '<div data-value="(foo)" class="bar">content</div>'
        result = _get_opening_tag(html, 'data-value="(foo)"')
        assert result == '<div data-value="(foo)" class="bar">'

    def test_marker_with_plus(self):
        html = '<span data-val="a+b" id="x">text</span>'
        result = _get_opening_tag(html, 'data-val="a+b"')
        assert result == '<span data-val="a+b" id="x">'


class TestSelfClosingTag:
    """Self-closing tags (e.g. <img />, <input />) are matched correctly."""

    def test_self_closing_tag(self):
        html = '<input data-testid="my-input" type="text" />'
        result = _get_opening_tag(html, 'data-testid="my-input"')
        assert result == '<input data-testid="my-input" type="text" />'

    def test_self_closing_tag_without_space(self):
        html = '<img data-testid="my-img" src="x.png"/>'
        result = _get_opening_tag(html, 'data-testid="my-img"')
        assert result == '<img data-testid="my-img" src="x.png"/>'
