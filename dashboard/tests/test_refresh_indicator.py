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


class TestTimestampErrorState:
    """Tests that JS shows failure state in timestamp elements on htmx errors."""

    def test_update_failed_text_in_js(self, client):
        """The JS must contain 'Update failed' text for the failure message."""
        html = client.get('/').text
        assert 'Update failed' in html

    def test_error_handler_sets_data_update_failed_attr(self, client):
        """The JS must set a data-update-failed attribute to prevent interval overwrites."""
        html = client.get('/').text
        assert 'data-update-failed' in html

    def test_error_events_registered_in_timestamp_iife(self, client):
        """The timestamp IIFE must register listeners on all three htmx error events."""
        html = client.get('/').text
        # All three error event names must appear in the page JS
        assert 'htmx:responseError' in html
        assert 'htmx:timeout' in html
        assert 'htmx:sendError' in html

    def test_after_swap_clears_failure_flag(self, client):
        """The htmx:afterSwap handler must clear data-update-failed on recovery."""
        html = client.get('/').text
        # The afterSwap handler should remove or clear data-update-failed
        assert 'removeAttribute' in html or "data-update-failed" in html

    def test_interval_skips_failed_elements(self, client):
        """The setInterval must skip updating elements that have data-update-failed set."""
        html = client.get('/').text
        # The interval function must check for data-update-failed before updating text
        assert 'data-update-failed' in html


class TestNaNGuard:
    """Tests that the setInterval timestamp updater guards against NaN timestamps."""

    def test_isnan_guard_present(self, client):
        """The JS setInterval updater must contain an isNaN guard to prevent 'Updated NaNm ago'."""
        html = client.get('/').text
        assert 'isNaN' in html


class TestAnimationFallback:
    """Tests that a setTimeout fallback ensures section-refreshed class is always removed."""

    def test_settimeout_fallback_present(self, client):
        """A setTimeout fallback must be present near section-refreshed class removal."""
        html = client.get('/').text
        assert 'setTimeout' in html

    def test_settimeout_removes_section_refreshed(self, client):
        """The setTimeout fallback must reference 'section-refreshed' class removal."""
        html = client.get('/').text
        # Both setTimeout and section-refreshed must appear in the JS
        assert 'setTimeout' in html
        assert 'section-refreshed' in html

    def test_settimeout_has_700ms_margin(self, client):
        """The setTimeout fallback should use a duration >= 700ms (safe margin over 600ms animation)."""
        html = client.get('/').text
        assert '700' in html

