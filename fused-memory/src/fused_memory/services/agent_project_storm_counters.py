"""Storm counters keyed per ``(agent label, project_id)``.

The keying policy shared by ``MemoryService``'s per-agent burst alarms: the
``update_memory`` content-amend storm (tasks 3088, 5263) and the
``ensure_entity_node`` mint storm (tasks 4932, 7011).

One counter per agent, because two independently-busy agents must not sum
into a false alarm. One counter per project as well, because recon stage ids
(``recon-stage-<stage>``) and curator ids are project-independent: pooled
across projects, one agent's writes elsewhere would inflate the count filed
into the crossing call's project queue, a number that project's own journal
rows could not reproduce.
"""

from __future__ import annotations

from typing import Any

from shared.storm_counter import KeyedStormCounters


class AgentProjectStormCounters:
    """A :class:`KeyedStormCounters` whose key is always ``(agent_label, project_id)``."""

    def __init__(self) -> None:
        self._counters: KeyedStormCounters[tuple[str, str]] = KeyedStormCounters()

    @property
    def tracked_agents(self) -> frozenset[str]:
        """The agent labels holding a storm window in any project.

        Agent ids are caller-supplied and unbounded, so evicting dormant ones
        is a real memory bound on a long-lived server.
        """
        return frozenset(agent for agent, _ in self._counters.tracked_keys)

    def record(
        self,
        agent_label: str,
        project_id: str,
        *,
        threshold: int,
        window_seconds: float,
        now: float,
    ) -> dict[str, Any] | None:
        """Record one event by *agent_label* in *project_id*; return its storm summary iff it fired.

        The summary's count spans *agent_label*'s events in *project_id* alone,
        and it is attributed to *agent_label*.
        """
        return self._counters.record(
            (agent_label, project_id),
            threshold=threshold,
            window_seconds=window_seconds,
            now=now,
            label=agent_label,
        )
