"""The legibility subsystem's one alarm: journal a finding, then post it as an escalate_info."""

from __future__ import annotations

import contextlib
import contextvars
import enum
import logging
import os
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from shared.mcp_session import SyncMcpSession

if TYPE_CHECKING:
    from legibility.config import LegibilityConfig

logger = logging.getLogger("legibility.alarm")

_TOOL_NAME = "escalate_info"
_CATEGORY = "infra_issue"
_SEVERITY = "info"
_POST_TIMEOUT_SECS = 10.0


class AlarmSource(enum.StrEnum):
    """Which legibility job raised the alarm; it names the escalation's task_id and agent_role."""

    CENSUS = "census"
    TRICKLE = "trickle"
    TRICKLE_HEALTH = "trickle-health"
    TRANSCRIPT_CHECK = "transcript-check"


@dataclass(frozen=True)
class AlarmProject:
    """The project an alarm is about, and the escalation server that hears it."""

    project_id: str
    escalation_port: int

    @classmethod
    def of(cls, cfg: LegibilityConfig) -> AlarmProject:
        return cls(project_id=cfg.project_id, escalation_port=cfg.escalation_port)

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.escalation_port}"


Poster = Callable[[str, Mapping[str, str]], object]
"""``poster(base_url, arguments)`` delivers one escalate_info and raises on failure."""


class RealPostRefusedUnderTest(RuntimeError):
    """A pytest process tried to post to an MCP endpoint it did not declare its own."""


_own_endpoint_declared: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "legibility_alarm_own_endpoint_declared", default=False,
)


@contextlib.contextmanager
def own_endpoint() -> Iterator[None]:
    """Declare that the MCP posts made inside this block target an endpoint the caller serves.

    The grant lasts exactly as long as the block, so it never covers a second
    endpoint the caller does not own.
    """
    token = _own_endpoint_declared.set(True)
    try:
        yield
    finally:
        _own_endpoint_declared.reset(token)


def refuse_real_post_under_test(url: str, tool_name: str) -> None:
    """Raise :class:`RealPostRefusedUnderTest` for an undeclared post from inside pytest.

    A test-minted escalation or task carries a real job's identity, so nothing
    downstream can tell it from a genuine one. No signal separates a test's
    own server from the production one on the same port, so entitlement is
    declared with :func:`own_endpoint`, never inferred.
    """
    if _own_endpoint_declared.get():
        return
    current_test = os.environ.get("PYTEST_CURRENT_TEST")
    if current_test is None:
        return
    raise RealPostRefusedUnderTest(
        f"legibility: refusing to post MCP tool {tool_name!r} to a real endpoint ({url}) "
        f"from inside a pytest process (PYTEST_CURRENT_TEST={current_test!r}). If this "
        f"caller serves or fakes its OWN endpoint, declare that around the call: "
        f"`with alarm.own_endpoint(): ...`."
    )


def _post_escalation(base_url: str, arguments: Mapping[str, str]) -> object:
    refuse_real_post_under_test(base_url, _TOOL_NAME)
    with SyncMcpSession(base_url, client_name="legibility-alarm", timeout=_POST_TIMEOUT_SECS) as session:
        return session.call_tool(_TOOL_NAME, arguments)


def raise_alarm(
    project: AlarmProject,
    source: AlarmSource,
    summary: str,
    detail: str,
    *,
    level: int,
    poster: Poster | None = None,
) -> bool:
    """Journal ``summary -- detail`` at *level*, then post it; True only if it landed.

    Never raises: a post that fails or is refused is one WARNING and False.
    """
    logger.log(level, "%s -- %s", summary, detail)
    arguments = {
        "task_id": f"legibility-{source}-{project.project_id}",
        "agent_role": f"legibility-{source}",
        "category": _CATEGORY,
        "severity": _SEVERITY,
        "summary": summary,
        "detail": detail,
    }
    try:
        reply = (poster or _post_escalation)(project.base_url, arguments)
    except Exception as exc:
        logger.warning(
            "legibility %s alarm did not reach %s: %s: %s",
            source, project.base_url, type(exc).__name__, exc,
        )
        return False
    if isinstance(reply, Mapping) and "error" in reply:
        logger.warning(
            "legibility %s alarm was refused by %s: %s", source, project.base_url, reply["error"],
        )
        return False
    return True


@contextlib.contextmanager
def fail_loud_boundary(
    source: AlarmSource, project: AlarmProject, *, poster: Poster | None = None,
) -> Iterator[None]:
    """Raise one ERROR alarm for an exception escaping the block, then re-raise it.

    The exit code stays the caller's to choose.
    """
    try:
        yield
    except Exception as exc:
        raise_alarm(
            project,
            source,
            f"legibility {source} run failed ({project.project_id}): {type(exc).__name__}: {exc}",
            "".join(traceback.format_exception(exc)),
            level=logging.ERROR,
            poster=poster,
        )
        raise
