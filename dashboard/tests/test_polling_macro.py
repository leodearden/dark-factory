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


class TestIndexHtmlUsesMacro:
    """Structural tests verifying index.html imports and uses polling_section."""

    _tmpl_dir = (
        Path(__file__).parent.parent / 'src' / 'dashboard' / 'templates'
    )

    def _index_content(self) -> str:
        return (self._tmpl_dir / 'index.html').read_text()

    def test_index_imports_polling_section(self):
        content = self._index_content()
        assert "from 'macros/polling_section.html' import polling_section" in content

    def test_index_has_call_block_for_orchestrators(self):
        content = self._index_content()
        assert "polling_section('orchestrators'" in content or 'polling_section("orchestrators"' in content

    def test_index_has_call_block_for_performance(self):
        content = self._index_content()
        assert "polling_section('performance'" in content or 'polling_section("performance"' in content

    def test_index_has_call_block_for_memory(self):
        content = self._index_content()
        # 'memory' appears in both 'memory' and 'memory-graphs'; check for standalone
        assert "polling_section('memory'" in content or 'polling_section("memory"' in content

    def test_index_has_call_block_for_memory_graphs(self):
        content = self._index_content()
        assert "polling_section('memory-graphs'" in content or 'polling_section("memory-graphs"' in content

    def test_index_has_call_block_for_recon(self):
        content = self._index_content()
        assert "polling_section('recon'" in content or 'polling_section("recon"' in content

    def test_index_has_five_call_blocks_total(self):
        content = self._index_content()
        # Count occurrences of the macro call pattern
        count = content.count('polling_section(')
        assert count >= 5, f'Expected at least 5 polling_section calls, got {count}'

    def test_orchestrators_call_passes_correct_hx_get(self):
        content = self._index_content()
        assert '/partials/orchestrators' in content

    def test_orchestrators_call_passes_correct_trigger(self):
        content = self._index_content()
        assert 'every 10s' in content

    def test_performance_call_passes_correct_timeout(self):
        content = self._index_content()
        assert '12000' in content

    def test_memory_graphs_call_passes_correct_trigger(self):
        content = self._index_content()
        assert 'every 60s' in content

    def test_recon_call_passes_correct_trigger(self):
        content = self._index_content()
        assert 'every 15s' in content


class TestEquivalenceJsCouplingAttributes:
    """Regression guard for JS coupling: data-section and data-updated-for
    must match for each of the 5 sections (base.html lines 46-90).

    These tests render each section via the macro and verify the critical
    attributes that the JavaScript depends on are correctly emitted.
    """

    # The 5 sections with their expected parameters from index.html
    _sections = [
        {
            'name': 'orchestrators',
            'hx_get': '/partials/orchestrators',
            'hx_trigger': 'load, every 10s',
            'timeout': 8000,
        },
        {
            'name': 'performance',
            'hx_get': '/partials/performance',
            'hx_trigger': 'load, every 30s',
            'timeout': 12000,
        },
        {
            'name': 'memory',
            'hx_get': '/partials/memory',
            'hx_trigger': 'load, every 10s',
            'timeout': 8000,
        },
        {
            'name': 'memory-graphs',
            'hx_get': '/partials/memory-graphs',
            'hx_trigger': 'load, every 60s',
            'timeout': 10000,
        },
        {
            'name': 'recon',
            'hx_get': '/partials/recon',
            'hx_trigger': 'load, every 15s',
            'timeout': 12000,
        },
    ]

    @pytest.mark.parametrize('section', _sections, ids=[s['name'] for s in _sections])
    def test_data_section_matches_data_updated_for(self, jinja_env, section):
        """data-section and data-updated-for must share the same name value."""
        html = render_polling_section(
            jinja_env,
            name=section['name'],
            hx_get=section['hx_get'],
            hx_trigger=section['hx_trigger'],
            hx_request_timeout=section['timeout'],
        )
        assert f'data-section="{section["name"]}"' in html
        assert f'data-updated-for="{section["name"]}"' in html

    @pytest.mark.parametrize('section', _sections, ids=[s['name'] for s in _sections])
    def test_hx_get_path_is_correct(self, jinja_env, section):
        """hx-get must point to the correct partial path."""
        html = render_polling_section(
            jinja_env,
            name=section['name'],
            hx_get=section['hx_get'],
            hx_trigger=section['hx_trigger'],
            hx_request_timeout=section['timeout'],
        )
        assert f'hx-get="{section["hx_get"]}"' in html

    @pytest.mark.parametrize('section', _sections, ids=[s['name'] for s in _sections])
    def test_timeout_value_is_in_hx_request(self, jinja_env, section):
        """The timeout value must appear in the hx-request attribute."""
        html = render_polling_section(
            jinja_env,
            name=section['name'],
            hx_get=section['hx_get'],
            hx_trigger=section['hx_trigger'],
            hx_request_timeout=section['timeout'],
        )
        assert str(section['timeout']) in html
        assert 'timeout' in html

    @pytest.mark.parametrize('section', _sections, ids=[s['name'] for s in _sections])
    def test_hx_swap_is_morph_inner_html(self, jinja_env, section):
        """hx-swap must always be morph:innerHTML for JS-safe DOM updates."""
        html = render_polling_section(
            jinja_env,
            name=section['name'],
            hx_get=section['hx_get'],
            hx_trigger=section['hx_trigger'],
            hx_request_timeout=section['timeout'],
        )
        assert 'hx-swap="morph:innerHTML"' in html
