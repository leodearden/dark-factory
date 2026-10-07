"""Tests for scripts/_config_reload_gate.py's public seam, driven in-process
over a real loopback socket against the stateful escalation MCP fake."""
from __future__ import annotations

import sys

import pytest
from _config_reload_gate import ReloadFailure, ReloadNotConfirmed, fetch_reload_report
from config_reload_script_fakes import (
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    reload_report,
)

COMMITTED_AS = "abc1234"
CONFIG_PATH = "/x/dark-factory-orchestrator.yaml"


def _refusal(port, *, config_path=CONFIG_PATH, **kwargs):
    """The ReloadNotConfirmed that fetch_reload_report raises against *port*."""
    with pytest.raises(ReloadNotConfirmed) as caught:
        fetch_reload_report(
            str(port), committed_as=COMMITTED_AS, config_path=config_path, **kwargs,
        )
    return caught.value


def test_a_healthy_reload_returns_the_unwrapped_report():
    report = reload_report(
        config_path=CONFIG_PATH,
        applied={"verify_env": {"old": {"K": "16"}, "new": {"K": "8"}}},
    )
    with FakeEscalationMcp(report) as server:
        fetched = fetch_reload_report(
            str(server.port), committed_as=COMMITTED_AS, config_path=CONFIG_PATH,
        )

    assert fetched == report


@pytest.mark.parametrize(
    "fault", list(TRANSPORT_FAULTS.values()), ids=list(TRANSPORT_FAULTS)
)
def test_every_transport_fault_is_no_reload_report(fault):
    with FakeEscalationMcp(reload_report(config_path="/x"), **fault) as server:
        refusal = _refusal(server.port)

    assert refusal.failure is ReloadFailure.NO_RELOAD_REPORT
    assert COMMITTED_AS in refusal.detail, refusal.detail


def test_a_dead_socket_is_no_reload_report():
    refusal = _refusal(ClosedPort().port)

    assert refusal.failure is ReloadFailure.NO_RELOAD_REPORT
    assert COMMITTED_AS in refusal.detail, refusal.detail


def test_a_reload_config_error_field_is_reload_error():
    """reload_config's own error is named as such, ahead of the uncommitted
    reload it also is."""
    with FakeEscalationMcp(reload_report(
        reloaded=False, error="load_config: while parsing a block mapping",
        config_path=CONFIG_PATH,
    )) as server:
        refusal = _refusal(server.port)

    assert refusal.failure is ReloadFailure.RELOAD_ERROR


# ---------------------------------------------------------------------------
# Whether the report is about the caller's own file
# ---------------------------------------------------------------------------

def test_a_reload_that_did_not_commit_is_reload_not_committed():
    """A failed reload rolls every leaf back, so nothing it reports is live."""
    with FakeEscalationMcp(reload_report(
        reloaded=False, error=None, config_path=CONFIG_PATH,
    )) as server:
        refusal = _refusal(server.port)

    assert refusal.failure is ReloadFailure.RELOAD_NOT_COMMITTED


@pytest.mark.parametrize(
    "reported",
    [
        "/elsewhere/dark-factory-orchestrator.yaml",
        None,
        # The reporting orchestrator's own ORCH_CONFIG_PATH, relative to ITS
        # cwd: resolved against the caller's it would match from this cwd.
        "dark-factory-orchestrator.yaml",
    ],
    ids=["another_file", "no_config_path", "relative_config_path"],
)
def test_a_reload_of_any_other_file_is_different_config_file(
    tmp_path, monkeypatch, reported,
):
    config = tmp_path / "dark-factory-orchestrator.yaml"
    monkeypatch.chdir(tmp_path)

    with FakeEscalationMcp(reload_report(config_path=reported)) as server:
        refusal = _refusal(server.port, config_path=str(config))

    assert refusal.failure is ReloadFailure.DIFFERENT_CONFIG_FILE
    assert str(config) in refusal.detail, refusal.detail


def test_the_callers_path_is_compared_by_realpath(tmp_path):
    """The caller may name its config through a symlink; the reloaded file is
    still the same file."""
    config = tmp_path / "dark-factory-orchestrator.yaml"
    config.write_text("")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(config)
    report = reload_report(config_path=str(config))

    with FakeEscalationMcp(report) as server:
        fetched = fetch_reload_report(
            str(server.port), committed_as=COMMITTED_AS, config_path=str(alias),
        )

    assert fetched == report


def test_without_a_config_path_the_reread_is_not_checked():
    """config_path=None asks only for the report: a caller that reads one
    knob's own disposition gets it whichever file was re-read."""
    report = reload_report(reloaded=False, config_path=None)

    with FakeEscalationMcp(report) as server:
        fetched = fetch_reload_report(
            str(server.port), committed_as=COMMITTED_AS, config_path=None,
        )

    assert fetched == report


def test_failure_tags_are_the_wire_vocabulary():
    """These strings are what a calling script prints in its JSON verdict, so
    renaming one breaks every consumer that branches on it."""
    assert {failure.value for failure in ReloadFailure} == {
        "transport_not_importable",
        "no_reload_report",
        "reload_reply_timed_out",
        "reload_error",
        "reload_not_committed",
        "different_config_file",
    }


def test_a_reply_slower_than_the_timeout_is_reply_timed_out():
    """The request was delivered, so the tool may have run: that is not a
    reload that never reached the tool."""
    with FakeEscalationMcp(
        reload_report(config_path="/x"), tool_call_delay=5.0,
    ) as server:
        refusal = _refusal(server.port, timeout=0.5)
        tools = server.called_tools()

    assert refusal.failure is ReloadFailure.REPLY_TIMED_OUT
    assert "reload_config" in tools, tools
    assert COMMITTED_AS in refusal.detail, refusal.detail


def test_an_interpreter_without_httpx_is_transport_not_importable(monkeypatch):
    """A host with pydantic but no httpx: the transport is not importable, and
    that is known before anything is sent."""
    monkeypatch.setitem(sys.modules, "httpx", None)

    with FakeEscalationMcp(reload_report(config_path="/x")) as server:
        refusal = _refusal(server.port)
        received = list(server.received)

    assert refusal.failure is ReloadFailure.TRANSPORT_NOT_IMPORTABLE
    assert sys.executable in refusal.detail, refusal.detail
    assert received == [], received
