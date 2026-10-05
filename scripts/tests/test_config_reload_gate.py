"""Tests for scripts/_config_reload_gate.py's public seam, driven in-process
over a real loopback socket against the stateful escalation MCP fake."""
from __future__ import annotations

import pytest
from _config_reload_gate import ReloadFailure, ReloadNotConfirmed, fetch_reload_report
from config_reload_script_fakes import (
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    reload_report,
)

COMMITTED_AS = "abc1234"


def _refusal(port, **kwargs):
    """The ReloadNotConfirmed that fetch_reload_report raises against *port*."""
    with pytest.raises(ReloadNotConfirmed) as caught:
        fetch_reload_report(str(port), committed_as=COMMITTED_AS, **kwargs)
    return caught.value


def test_a_healthy_reload_returns_the_unwrapped_report():
    report = reload_report(
        config_path="/x/y.yaml",
        applied={"verify_env": {"old": {"K": "16"}, "new": {"K": "8"}}},
    )
    with FakeEscalationMcp(report) as server:
        fetched = fetch_reload_report(str(server.port), committed_as=COMMITTED_AS)

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
    with FakeEscalationMcp(reload_report(
        reloaded=False, error="load_config: while parsing a block mapping",
    )) as server:
        refusal = _refusal(server.port)

    assert refusal.failure is ReloadFailure.RELOAD_ERROR


def test_failure_tags_are_the_wire_vocabulary():
    """These strings are what a calling script prints in its JSON verdict, so
    renaming one breaks every consumer that branches on it."""
    assert {failure.value for failure in ReloadFailure} == {
        "transport_not_importable",
        "no_reload_report",
        "reload_reply_timed_out",
        "reload_error",
    }
