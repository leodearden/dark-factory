"""``shared.cli_proxy.proxy_endpoint`` reads the proxy's client file the way ``claude-proxy`` sources it.

``shared.claude_endpoint.ClaudeEndpoint`` refuses an endpoint that would silently fall back to a login.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.claude_endpoint import ClaudeEndpoint
from shared.cli_proxy import ProxyClientFileInvalid, proxy_endpoint

KEY = 'cpa-' + 'k' * 44


def _client_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'client.env'
    path.write_text(text, encoding='utf-8')
    return path


def test_the_wrappers_export_lines_name_the_endpoint(tmp_path):
    path = _client_file(
        tmp_path, f'export ANTHROPIC_BASE_URL=http://127.0.0.1:8317\nexport ANTHROPIC_AUTH_TOKEN={KEY}\n',
    )

    assert proxy_endpoint(path) == ClaudeEndpoint(base_url='http://127.0.0.1:8317', auth_token=KEY)


def test_no_client_file_is_no_proxy(tmp_path):
    assert proxy_endpoint(tmp_path / 'absent.env') is None


@pytest.mark.parametrize(
    ('text', 'missing'),
    [
        pytest.param(f'export ANTHROPIC_AUTH_TOKEN={KEY}\n', 'ANTHROPIC_BASE_URL', id='no-base-url'),
        pytest.param('export ANTHROPIC_BASE_URL=http://127.0.0.1:8317\n', 'ANTHROPIC_AUTH_TOKEN', id='no-key'),
        pytest.param('export ANTHROPIC_BASE_URL=\nexport ANTHROPIC_AUTH_TOKEN=\n', 'ANTHROPIC_BASE_URL', id='empty'),
    ],
)
def test_a_client_file_without_both_variables_is_refused_by_name(tmp_path, text, missing):
    with pytest.raises(ProxyClientFileInvalid, match=missing):
        proxy_endpoint(_client_file(tmp_path, text))


def test_the_key_stays_out_of_the_repr():
    endpoint = ClaudeEndpoint(base_url='http://127.0.0.1:8317', auth_token=KEY)

    assert KEY not in repr(endpoint)
    assert endpoint.env() == {'ANTHROPIC_BASE_URL': 'http://127.0.0.1:8317', 'ANTHROPIC_AUTH_TOKEN': KEY}


@pytest.mark.parametrize(
    ('base_url', 'auth_token'),
    [
        pytest.param('', KEY, id='empty-url'),
        pytest.param('127.0.0.1:8317', KEY, id='no-scheme'),
        pytest.param('http://127.0.0.1:8317', '', id='empty-key'),
    ],
)
def test_an_endpoint_that_would_fall_back_to_a_login_is_refused(base_url, auth_token):
    with pytest.raises(ValueError, match='endpoint'):
        ClaudeEndpoint(base_url=base_url, auth_token=auth_token)
