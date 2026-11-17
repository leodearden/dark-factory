"""The one node-binary helper: the binary when on PATH, a skip locally, a failure under CI."""

from __future__ import annotations

import shutil

import pytest
from _dashboard_helpers import node_path


def test_node_on_path_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, 'which', lambda name: '/opt/node/bin/node')

    assert node_path() == '/opt/node/bin/node'


def test_node_absent_outside_ci_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    monkeypatch.delenv('CI', raising=False)

    with pytest.raises(pytest.skip.Exception):
        node_path()


def test_node_absent_under_ci_fails_naming_node_and_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    monkeypatch.setenv('CI', '1')

    with pytest.raises(pytest.fail.Exception) as excinfo:
        node_path()

    message = str(excinfo.value)
    assert 'node' in message
    assert 'CI' in message
