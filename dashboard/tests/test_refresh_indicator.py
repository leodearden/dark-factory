"""Tests for the per-section last-updated timestamp and refresh indicator feature."""

from __future__ import annotations


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
        assert 'text-gray-500' in html

    def test_timestamp_elements_are_aria_hidden(self, client):
        html = client.get('/').text
        assert 'aria-hidden="true"' in html


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
