"""Wiring tests for the Scheduler tab UI (frontend).

Tests parse JSX/CSS source files as text and assert structural contracts
(CSS width values, export names, component patterns). Follows the idiom
established in test_tab_curator.py and test_index_html.py.

Each RED test is added before its corresponding GREEN implementation step.
Actual rendering must be visually verified — these are source-structure
assertions only.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def styles_css_body(_client):
    return _client.get('/static/redux/styles.css').text


@pytest.fixture(scope='module')
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def tab_scheduler_jsx_body(_client):
    return _client.get('/static/redux/tab_scheduler.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text
