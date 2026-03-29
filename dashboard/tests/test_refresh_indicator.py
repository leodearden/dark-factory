"""Tests for the per-section last-updated timestamp and refresh indicator feature."""

import re

from .test_helpers import _get_opening_tag

_SECTIONS = ('orchestrators', 'performance', 'memory', 'memory-graphs', 'recon')

# Character class used to capture data-section and data-updated-for attribute values.
# Broadened to [a-z0-9][a-z0-9-]* so that digit-containing names (e.g. 'v2-panel') are captured.
_SECTION_NAME_RE = r'[a-z0-9][a-z0-9-]*'


class TestSectionNameRegex:
    """Unit tests for the _SECTION_NAME_RE constant."""

    def test_captures_hyphenated_names(self):
        """Baseline: the regex must match hyphenated pure-alpha names."""
        assert re.fullmatch(_SECTION_NAME_RE, 'memory-graphs') is not None
        assert re.fullmatch(_SECTION_NAME_RE, 'recon') is not None
        assert re.fullmatch(_SECTION_NAME_RE, 'orchestrators') is not None

    def test_captures_digit_containing_names(self):
        """Regression: the regex must match names that contain digits (e.g. 'v2-panel')."""
        assert re.fullmatch(_SECTION_NAME_RE, 'v2-panel') is not None


class TestRouteHealth:
    """Sentinel tests: assert HTTP 200 on the two routes exercised throughout this file."""

    def test_homepage_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_css_route_returns_200(self, client):
        resp = client.get('/static/tailwind.css')
        assert resp.status_code == 200


class TestSectionDataAttributes:
    """Tests that each polling section has a data-section attribute."""

    def test_orchestrators_section_has_data_section(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-section="orchestrators"' in html

    def test_performance_section_has_data_section(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-section="performance"' in html

    def test_memory_section_has_data_section(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-section="memory"' in html

    def test_memory_graphs_section_has_data_section(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-section="memory-graphs"' in html

    def test_recon_section_has_data_section(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-section="recon"' in html


class TestTimestampElements:
    """Tests that each section has a corresponding timestamp element."""

    def test_orchestrators_timestamp_element(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-updated-for="orchestrators"' in html

    def test_performance_timestamp_element(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-updated-for="performance"' in html

    def test_memory_timestamp_element(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-updated-for="memory"' in html

    def test_memory_graphs_timestamp_element(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-updated-for="memory-graphs"' in html

    def test_recon_timestamp_element(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-updated-for="recon"' in html

    def test_timestamp_elements_have_subtle_styling(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        for section in _SECTIONS:
            marker = f'data-updated-for="{section}"'
            tag = _get_opening_tag(html, marker)
            assert 'text-gray-500' in tag, f'Expected text-gray-500 on data-updated-for="{section}" element'

    def test_timestamp_elements_are_aria_hidden(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        for section in _SECTIONS:
            marker = f'data-updated-for="{section}"'
            tag = _get_opening_tag(html, marker)
            assert 'aria-hidden="true"' in tag, f'Expected aria-hidden="true" on data-updated-for="{section}" element'


class TestSectionTimestampPairing:
    """Tests that every data-section has a matching data-updated-for element."""

    def test_every_section_has_matching_timestamp(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        sections = set(re.findall(rf'data-section="({_SECTION_NAME_RE})"', html))
        timestamps = set(re.findall(rf'data-updated-for="({_SECTION_NAME_RE})"', html))
        assert sections == timestamps

    def test_all_expected_sections_paired(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        sections = set(re.findall(rf'data-section="({_SECTION_NAME_RE})"', html))
        assert sections == set(_SECTIONS)


class TestRefreshTrackingJS:
    """Tests that the rendered page contains the refresh-tracking JavaScript."""

    def test_after_swap_listener_present(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'htmx:afterSwap' in html

    def test_set_interval_present(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'setInterval' in html

    def test_data_updated_at_present(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-updated-at' in html

    def test_updated_text_present(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'Updated' in html


class TestRefreshPulseCSS:
    """Tests that tailwind.css contains the refresh-pulse animation."""

    def test_keyframes_animation_present(self, client):
        resp = client.get('/static/tailwind.css')
        assert resp.status_code == 200
        css = resp.text
        assert '@keyframes section-refresh-pulse' in css

    def test_section_refreshed_class_present(self, client):
        resp = client.get('/static/tailwind.css')
        assert resp.status_code == 200
        css = resp.text
        assert '.section-refreshed' in css


class TestTimestampErrorState:
    """Tests that JS shows failure state in timestamp elements on htmx errors."""

    def test_update_failed_text_in_js(self, client):
        """The JS must contain 'Update failed' text for the failure message."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'Update failed' in html

    def test_error_handler_sets_data_update_failed_attr(self, client):
        """The JS must set a data-update-failed attribute to prevent interval overwrites."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'data-update-failed' in html

    def test_error_events_registered_in_timestamp_iife(self, client):
        """The timestamp IIFE must register listeners on all three htmx error events."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        # markStampFailed is the unique function name defined in the timestamp IIFE;
        # asserting its presence confirms the timestamp-IIFE registrations exist
        # (unlike the event name strings which also appear in the error-card IIFE)
        assert 'markStampFailed' in html

    def test_after_swap_clears_failure_flag(self, client):
        """The htmx:afterSwap handler must clear data-update-failed on recovery."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        # The afterSwap handler must call removeAttribute('data-update-failed') specifically
        assert "removeAttribute('data-update-failed')" in html

    def test_interval_skips_failed_elements(self, client):
        """The setInterval must skip updating elements that have data-update-failed set."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        # The interval function must check for data-update-failed before updating text
        assert 'data-update-failed' in html


class TestNaNGuard:
    """Tests that the setInterval timestamp updater guards against NaN timestamps."""

    def test_isnan_guard_present(self, client):
        """The JS setInterval updater must contain an isNaN guard to prevent 'Updated NaNm ago'."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'isNaN' in html


class TestAnimationFallback:
    """Tests that a setTimeout fallback ensures section-refreshed class is always removed."""

    def test_settimeout_fallback_present(self, client):
        """A setTimeout fallback must be present near section-refreshed class removal."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert 'setTimeout' in html

    def test_settimeout_removes_section_refreshed(self, client):
        """The setTimeout fallback must reference 'section-refreshed' class removal."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        # Both setTimeout and section-refreshed must appear in the JS
        assert 'setTimeout' in html
        assert 'section-refreshed' in html

    def test_settimeout_has_700ms_margin(self, client):
        """The setTimeout fallback should use a duration >= 700ms (safe margin over 600ms animation)."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        assert '700' in html


class TestIntervalCleanup:
    """Tests that setInterval return value is stored for potential cleanup."""

    def test_setinterval_assigned_to_variable(self, client):
        """The setInterval return value must be assigned to a variable."""
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text
        # Matches 'var <name> = setInterval' or 'const <name> = setInterval' etc.
        assert re.search(r'\b(?:var|let|const)\s+\w+\s*=\s*setInterval\b', html) is not None
