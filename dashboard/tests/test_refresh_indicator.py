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
