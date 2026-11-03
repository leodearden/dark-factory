"""A fused-memory curator double with the ``Scheduler.dispatch_tool`` signature.

Bind an instance wherever production code holds a bound
``orchestrator/src/orchestrator/scheduler.py::Scheduler.dispatch_tool``.  It
answers ``submit_task`` and ``resolve_ticket`` with JSON-RPC bodies shaped like
``orchestrator/tests/_orch_helpers.py::HermeticMcpSession._envelope``, records
every call, and raises ``NotImplementedError`` for any other tool so an
unanticipated call fails the test loudly.
"""

from __future__ import annotations

import json
from typing import Any

UNDECIDED_TICKET: dict[str, Any] = {'status': 'failed', 'reason': 'timeout', 'task_id': None}


class FakeCurator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._submit_payload: dict[str, Any] | None = None
        self._submit_raises: Exception | None = None
        self._tickets: dict[str, dict[str, Any]] = {}
        self._issued = 0
        self._rpc_id = 0

    def fail_submits(self, error: str = 'curator closed', error_type: str = 'CuratorClosed') -> None:
        self._submit_payload = {'error': error, 'error_type': error_type}

    def combine_submits(self, task_id: str) -> None:
        self._submit_payload = {'task_id': task_id, 'combined': True}

    def raise_on_submit(self, exc: Exception | None = None) -> None:
        self._submit_raises = exc or ConnectionError('fused-memory unreachable')

    def set_ticket(self, ticket: str, answer: dict[str, Any]) -> None:
        self._tickets[ticket] = dict(answer)

    def calls_named(self, name: str) -> list[dict]:
        return [arguments for called, arguments in self.calls if called == name]

    async def __call__(self, name: str, arguments: dict, *, timeout: float = 15) -> dict:
        self.calls.append((name, dict(arguments)))
        if name == 'submit_task':
            return self._submit()
        if name == 'resolve_ticket':
            return self._envelope(self._tickets.get(arguments.get('ticket', ''), UNDECIDED_TICKET))
        raise NotImplementedError(f'FakeCurator does not answer {name!r}')

    def _submit(self) -> dict:
        if self._submit_raises is not None:
            raise self._submit_raises
        if self._submit_payload is not None:
            return self._envelope(self._submit_payload)
        self._issued += 1
        return self._envelope({'ticket': f'tkt_{self._issued}'})

    def _envelope(self, payload: dict[str, Any]) -> dict:
        self._rpc_id += 1
        return {
            'jsonrpc': '2.0',
            'id': self._rpc_id,
            'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]},
        }
