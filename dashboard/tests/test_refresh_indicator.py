"""Tests for the per-section last-updated timestamp and refresh indicator feature."""

import re

_SECTIONS = ('orchestrators', 'performance', 'memory', 'memory-graphs', 'recon')


class TestSectionDataAttributes:
    """Tests that each polling section has a data-section attribute."""

    def test_orchestrators_section_has_data_section(self, client):
        html = client.get('/').text
        assert 'data-section="orchestrators"' in html

    def test_performance_section_has_data_section(self, client):
        html = client.get('/').text
        assert 'data-section="performance"' in html

    def test_memory_section_has_data_section(self, client):
        html = client.get('/').text
        assert 'data-section="memory"' in html

    def test_memory_graphs_section_has_data_section(self, client):
        html = client.get('/').text
        assert 'data-section="memory-graphs"' in html

    def test_recon_section_has_data_section(self, client):
        html = client.get('/').text
        assert 'data-section="recon"' in html


class TestTimestampElements:
    """Tests that each section has a corresponding timestamp element."""

    def test_orchestrators_timestamp_element(self, client):
        html = client.get('/').text
        assert 'data-updated-for="orchestrators"' in html

    def test_performance_timestamp_element(self, client):
        html = client.get('/').text
        assert 'data-updated-for="performance"' in html

    def test_memory_timestamp_element(self, client):
        html = client.get('/').text
        assert 'data-updated-for="memory"' in html

    def test_memory_graphs_timestamp_element(self, client):
        html = client.get('/').text
        assert 'data-updated-for="memory-graphs"' in html

    def test_recon_timestamp_element(self, client):
        html = client.get('/').text
        assert 'data-updated-for="recon"' in html

    def test_timestamp_elements_have_subtle_styling(self, client):
        html = client.get('/').text
        for section in _SECTIONS:
            marker = f'data-updated-for="{section}"'
            idx = html.index(marker)
            tag_start = html.rfind('<', 0, idx)
            tag_end = html.index('>', idx)
            tag = html[tag_start:tag_end + 1]
            assert 'text-gray-500' in tag, f'Expected text-gray-500 on data-updated-for="{section}" element'

    def test_timestamp_elements_are_aria_hidden(self, client):
        html = client.get('/').text
        for section in _SECTIONS:
            marker = f'data-updated-for="{section}"'
            idx = html.index(marker)
            tag_start = html.rfind('<', 0, idx)
            tag_end = html.index('>', idx)
            tag = html[tag_start:tag_end + 1]
            assert 'aria-hidden="true"' in tag, f'Expected aria-hidden="true" on data-updated-for="{section}" element'


class TestSectionTimestampPairing:
    """Tests that every data-section has a matching data-updated-for element."""

    def test_every_section_has_matching_timestamp(self, client):
        html = client.get('/').text
        sections = set(re.findall(r'data-section="([a-z][a-z-]*)"', html))
        timestamps = set(re.findall(r'data-updated-for="([a-z][a-z-]*)"', html))
        assert sections == timestamps

    def test_all_expected_sections_paired(self, client):
        html = client.get('/').text
        sections = set(re.findall(r'data-section="([a-z][a-z-]*)"', html))
        assert sections == set(_SECTIONS)


class TestRefreshTrackingJS:
    """Tests that the rendered page contains the refresh-tracking JavaScript."""

    def test_after_swap_listener_present(self, client):
        html = client.get('/').text
        assert 'htmx:afterSwap' in html

    def test_set_interval_present(self, client):
        html = client.get('/').text
        assert 'setInterval' in html

    def test_data_updated_at_present(self, client):
        html = client.get('/').text
        assert 'data-updated-at' in html

    def test_updated_text_present(self, client):
        html = client.get('/').text
        assert 'Updated' in html


class TestRefreshPulseCSS:
    """Tests that tailwind.css contains the refresh-pulse animation."""

    def test_keyframes_animation_present(self, client):
        css = client.get('/static/tailwind.css').text
        assert '@keyframes section-refresh-pulse' in css

    def test_section_refreshed_class_present(self, client):
        css = client.get('/static/tailwind.css').text
        assert '.section-refreshed' in css

