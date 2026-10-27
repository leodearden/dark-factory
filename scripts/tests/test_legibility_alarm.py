"""Tests for scripts/legibility/alarm.py, the legibility subsystem's one
escalate_info poster.

Imported in the package form (`from legibility import alarm`), the form every
importer uses, so the `own_endpoint` ContextVar these tests set is the one the
poster reads. Real posts go to `shared.testing_mcp_stub.StubMcpServer` over a
real socket.
"""
from __future__ import annotations

import logging

import pytest
from legibility import alarm, config
from shared.testing_mcp_stub import StubMcpServer

_ALARM_LOGGER = "legibility.alarm"


def _port(server: StubMcpServer) -> int:
    return int(server.url.rsplit(":", 1)[1])


def _project(port: int = 8103) -> alarm.AlarmProject:
    return alarm.AlarmProject(project_id="p1", escalation_port=port)


def _expected_arguments(source: alarm.AlarmSource) -> dict[str, str]:
    return {
        "task_id": f"legibility-{source}-p1",
        "agent_role": f"legibility-{source}",
        "category": "infra_issue",
        "severity": "info",
        "summary": "S",
        "detail": "D",
    }


class _Recorder:
    def __init__(self, reply: object = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._reply = reply

    def __call__(self, base_url, arguments):
        self.calls.append((base_url, dict(arguments)))
        return self._reply


def _raising_poster(base_url, arguments):
    raise ConnectionError("escalation server gone")


def _delivery_warnings(caplog) -> list[logging.LogRecord]:
    """The alarm's WARNINGs about its post, leaving out a WARNING-level journal line."""
    return [
        r for r in caplog.records
        if r.name == _ALARM_LOGGER and r.levelno == logging.WARNING and r.getMessage() != "S -- D"
    ]


def _tool_calls(server: StubMcpServer) -> list[dict]:
    return [
        r.body for r in server.requests
        if isinstance(r.body, dict) and r.body.get("method") == "tools/call"
    ]


def test_the_sources_are_the_four_legibility_alarm_labels():
    assert [s.value for s in alarm.AlarmSource] == [
        "census", "trickle", "trickle-health", "transcript-check",
    ]


@pytest.mark.parametrize("source", list(alarm.AlarmSource))
def test_raise_alarm_posts_escalate_info_arguments_to_the_project_base_url(source):
    recorder = _Recorder()

    landed = alarm.raise_alarm(
        _project(8123), source, "S", "D", level=logging.WARNING, poster=recorder,
    )

    assert landed is True
    assert recorder.calls == [("http://localhost:8123", _expected_arguments(source))]


def test_alarm_project_of_reads_the_config_identity_and_port(tmp_path):
    cfg = config.LegibilityConfig(
        project_id="proj-x",
        project_root=str(tmp_path),
        escalation_port=8777,
        cwd_prefixes=[str(tmp_path)],
    )

    project = alarm.AlarmProject.of(cfg)

    assert project == alarm.AlarmProject(project_id="proj-x", escalation_port=8777)
    assert project.base_url == "http://localhost:8777"


@pytest.mark.parametrize("level", [logging.WARNING, logging.ERROR])
def test_the_alarm_is_journaled_before_the_poster_runs(caplog, level):
    seen_before_post: list[list[tuple[int, str]]] = []

    def _poster(base_url, arguments):
        seen_before_post.append([
            (r.levelno, r.getMessage()) for r in caplog.records if r.name == _ALARM_LOGGER
        ])

    with caplog.at_level(logging.DEBUG, logger=_ALARM_LOGGER):
        alarm.raise_alarm(_project(), alarm.AlarmSource.CENSUS, "S", "D", level=level, poster=_poster)

    assert seen_before_post == [[(level, "S -- D")]]


def test_the_alarm_is_journaled_even_when_the_poster_raises(caplog):
    with caplog.at_level(logging.DEBUG, logger=_ALARM_LOGGER):
        alarm.raise_alarm(
            _project(), alarm.AlarmSource.TRICKLE, "S", "D", level=logging.ERROR, poster=_raising_poster,
        )

    assert (logging.ERROR, "S -- D") in [(r.levelno, r.getMessage()) for r in caplog.records]


def test_a_raising_poster_is_one_warning_and_false(caplog):
    with caplog.at_level(logging.DEBUG, logger=_ALARM_LOGGER):
        landed = alarm.raise_alarm(
            _project(), alarm.AlarmSource.TRICKLE, "S", "D", level=logging.ERROR, poster=_raising_poster,
        )

    assert landed is False
    [warning] = _delivery_warnings(caplog)
    assert "ConnectionError" in warning.getMessage()
    assert "escalation server gone" in warning.getMessage()


def test_an_error_reply_is_not_landed(caplog):
    recorder = _Recorder(reply={"error": "invalid severity"})

    with caplog.at_level(logging.DEBUG, logger=_ALARM_LOGGER):
        landed = alarm.raise_alarm(
            _project(), alarm.AlarmSource.CENSUS, "S", "D", level=logging.WARNING, poster=recorder,
        )

    assert landed is False
    [warning] = _delivery_warnings(caplog)
    assert "invalid severity" in warning.getMessage()


@pytest.mark.parametrize("reply", [None, {"id": "esc-1"}])
def test_a_reply_without_an_error_is_landed(reply):
    landed = alarm.raise_alarm(
        _project(), alarm.AlarmSource.CENSUS, "S", "D", level=logging.WARNING, poster=_Recorder(reply),
    )

    assert landed is True


def test_a_down_escalation_server_is_one_warning_and_false(caplog):
    with StubMcpServer(stateful=True) as closed_server:
        port = _port(closed_server)

    with caplog.at_level(logging.DEBUG, logger=_ALARM_LOGGER), alarm.own_endpoint():
        landed = alarm.raise_alarm(
            _project(port), alarm.AlarmSource.CENSUS, "S", "D", level=logging.ERROR,
        )

    assert landed is False
    [warning] = _delivery_warnings(caplog)
    assert "McpCallError" in warning.getMessage()


@pytest.mark.parametrize("source", list(alarm.AlarmSource))
def test_under_pytest_the_default_poster_refuses_a_real_post(caplog, source):
    with StubMcpServer(stateful=True) as server:
        with caplog.at_level(logging.DEBUG, logger=_ALARM_LOGGER):
            landed = alarm.raise_alarm(
                _project(_port(server)), source, "S", "D", level=logging.WARNING,
            )
        requests = server.requests

    assert landed is False
    assert requests == ()
    [warning] = _delivery_warnings(caplog)
    assert "RealPostRefusedUnderTest" in warning.getMessage()
    assert "own_endpoint" in warning.getMessage()


def test_inside_own_endpoint_the_default_poster_lands_one_escalate_info():
    payload = {"id": "esc-1", "status": "queued"}
    with StubMcpServer(stateful=True, payload=payload) as server:
        with alarm.own_endpoint():
            landed = alarm.raise_alarm(
                _project(_port(server)), alarm.AlarmSource.TRICKLE, "S", "D", level=logging.WARNING,
            )
        tool_calls = _tool_calls(server)

    assert landed is True
    assert [call["params"] for call in tool_calls] == [
        {"name": "escalate_info", "arguments": _expected_arguments(alarm.AlarmSource.TRICKLE)},
    ]


def test_the_own_endpoint_grant_does_not_outlive_its_block():
    url = "http://localhost:8103"

    with alarm.own_endpoint():
        assert alarm.refuse_real_post_under_test(url, "escalate_info") is None
    with pytest.raises(alarm.RealPostRefusedUnderTest):
        alarm.refuse_real_post_under_test(url, "escalate_info")

    with pytest.raises(RuntimeError, match="entitled block failed"), alarm.own_endpoint():
        raise RuntimeError("entitled block failed")
    with pytest.raises(alarm.RealPostRefusedUnderTest):
        alarm.refuse_real_post_under_test(url, "escalate_info")


def test_the_refusal_names_the_tool_the_url_and_the_hatch():
    with pytest.raises(alarm.RealPostRefusedUnderTest) as excinfo:
        alarm.refuse_real_post_under_test("http://localhost:8103", "escalate_info")

    message = str(excinfo.value)
    assert "escalate_info" in message
    assert "http://localhost:8103" in message
    assert "own_endpoint" in message


def test_outside_pytest_the_guard_stands_aside(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST")

    assert alarm.refuse_real_post_under_test("http://localhost:8103", "escalate_info") is None
