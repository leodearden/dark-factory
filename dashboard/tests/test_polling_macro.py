"""Unit tests for Jinja2 polling_section macro in macros/polling_section.html.

These tests render the macro in isolation via a Jinja2 FileSystemLoader,
without requiring the full FastAPI stack.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

# Path to the templates directory (same root as app.py uses)
_TEMPLATES_DIR = (
    Path(__file__).parent.parent / 'src' / 'dashboard' / 'templates'
)


@pytest.fixture(scope='module')
def jinja_env() -> jinja2.Environment:
    """Jinja2 Environment pointing at the dashboard templates directory."""
    loader = jinja2.FileSystemLoader(str(_TEMPLATES_DIR))
    return jinja2.Environment(loader=loader, autoescape=True)


def render_polling_section(
    env: jinja2.Environment,
    name: str,
    hx_get: str,
    hx_trigger: str,
    hx_request_timeout: int,
    caller_content: str = '<p>skeleton content</p>',
) -> str:
    """Render the polling_section macro with a caller block and return HTML."""
    source = (
        "{% from 'macros/polling_section.html' import polling_section %}"
        "{% call polling_section(name, hx_get, hx_trigger, hx_request_timeout) %}"
        + caller_content
        + "{% endcall %}"
    )
    tmpl = env.from_string(source)
    return tmpl.render(
        name=name,
        hx_get=hx_get,
        hx_trigger=hx_trigger,
        hx_request_timeout=hx_request_timeout,
    )


class TestPollingSectionAttributes:
    """Tests for the section element attributes produced by polling_section."""

    def test_section_has_data_section_equal_name(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert 'data-section="orchestrators"' in html

    def test_section_has_hx_get_attribute(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='performance',
            hx_get='/partials/performance',
            hx_trigger='load, every 30s',
            hx_request_timeout=12000,
        )
        assert 'hx-get="/partials/performance"' in html

    def test_section_has_hx_trigger_attribute(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='memory',
            hx_get='/partials/memory',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert 'hx-trigger="load, every 10s"' in html

    def test_section_has_hx_request_with_timeout_json(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='memory-graphs',
            hx_get='/partials/memory-graphs',
            hx_trigger='load, every 60s',
            hx_request_timeout=10000,
        )
        assert '10000' in html
        assert 'timeout' in html

    def test_section_has_hx_swap_morph_inner_html(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='recon',
            hx_get='/partials/recon',
            hx_trigger='load, every 15s',
            hx_request_timeout=12000,
        )
        assert 'hx-swap="morph:innerHTML"' in html

    def test_section_has_aria_live_polite(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert 'aria-live="polite"' in html


class TestPollingSectionUpdatedForDiv:
    """Tests for the data-updated-for timestamp div."""

    def test_data_updated_for_has_matching_name(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert 'data-updated-for="orchestrators"' in html

    def test_data_updated_for_name_matches_data_section_name(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='recon',
            hx_get='/partials/recon',
            hx_trigger='load, every 15s',
            hx_request_timeout=12000,
        )
        assert 'data-section="recon"' in html
        assert 'data-updated-for="recon"' in html

    def test_data_updated_for_div_has_aria_hidden_true(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='memory',
            hx_get='/partials/memory',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert 'aria-hidden="true"' in html

    def test_data_updated_for_div_has_text_xs_class(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='performance',
            hx_get='/partials/performance',
            hx_trigger='load, every 30s',
            hx_request_timeout=12000,
        )
        assert 'text-xs' in html

    def test_data_updated_for_div_has_text_gray_500_class(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='performance',
            hx_get='/partials/performance',
            hx_trigger='load, every 30s',
            hx_request_timeout=12000,
        )
        assert 'text-gray-500' in html

    def test_data_updated_for_div_has_mt_1_class(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='recon',
            hx_get='/partials/recon',
            hx_trigger='load, every 15s',
            hx_request_timeout=12000,
        )
        assert 'mt-1' in html

    def test_data_updated_for_div_has_h_4_class(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert 'h-4' in html


class TestPollingSectionStructure:
    """Tests for the overall HTML structure of the polling_section macro."""

    def test_outer_wrapper_div_is_present(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert '<div>' in html or '<div ' in html

    def test_caller_block_content_is_rendered_inside_section(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
            caller_content='<p class="unique-skeleton-content">unique text</p>',
        )
        assert 'unique-skeleton-content' in html
        assert 'unique text' in html

    def test_caller_content_appears_before_updated_for_div(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='orchestrators',
            hx_get='/partials/orchestrators',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
            caller_content='<p class="skeleton-marker">loading</p>',
        )
        skeleton_pos = html.find('skeleton-marker')
        updated_for_pos = html.find('data-updated-for')
        assert skeleton_pos != -1
        assert updated_for_pos != -1
        assert skeleton_pos < updated_for_pos

    def test_section_element_is_rendered(self, jinja_env):
        html = render_polling_section(
            jinja_env,
            name='memory',
            hx_get='/partials/memory',
            hx_trigger='load, every 10s',
            hx_request_timeout=8000,
        )
        assert '<section' in html
        assert '</section>' in html
