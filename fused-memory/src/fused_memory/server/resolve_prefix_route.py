"""POST /resolve-prefix: the uuid-prefix resolver over HTTP (PRD C5).

A process that cannot import fused_memory (the escalation servers' resolver
client, leaf δ) asks here. 200 carries the Resolution itself; 400 names the
malformed request field; 503 names the store that could not be read, because
an unchecked prefix is never a 200 ``none`` (INV-11).

This module is the one home of the resolver's wire vocabulary:
``resolution_body`` and ``resolver_unavailable_body`` are shared with the read
tools' prefix arms in ``server/tools.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from shared.uuid_prefix_guard import PrefixResolver, Resolution, ResolverUnavailable
from starlette.requests import Request
from starlette.responses import JSONResponse

from fused_memory.services.id_resolver import RESOLVABLE_PREFIX_GRAMMAR, is_resolvable_prefix
from fused_memory.utils.validation import (
    PathShapedProjectIdError,
    canonicalize_project_id,
    validate_known_project_id,
    validate_project_id,
)

logger = logging.getLogger(__name__)

RESOLVE_PREFIX_PATH = '/resolve-prefix'
RESOLVER_UNAVAILABLE_ERROR_TYPE = 'resolver_unavailable'


def resolution_body(resolution: Resolution) -> dict[str, Any]:
    """Exactly the Resolution's fields, so a client decodes it without a parser."""
    return {
        'outcome': resolution.outcome,
        'candidates': [candidate._asdict() for candidate in resolution.candidates],
    }


def resolver_unavailable_body(exc: ResolverUnavailable) -> dict[str, Any]:
    """The outage shape: which store failed, never an outcome."""
    return {'error': str(exc), 'error_type': RESOLVER_UNAVAILABLE_ERROR_TYPE, 'store': exc.store}


class _InvalidRequest(Exception):
    def __init__(self, field: str, error: str) -> None:
        super().__init__(error)
        self.field = field


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise _InvalidRequest('body', f'request body is not JSON: {exc}') from exc
    if not isinstance(body, dict):
        raise _InvalidRequest('body', 'request body must be a JSON object carrying project_id and prefix')
    return body


def _known_project_id(raw: object, known_projects: dict[str, str] | None) -> str:
    if not isinstance(raw, str):
        raise _InvalidRequest('project_id', 'project_id must be a string')
    try:
        project_id = canonicalize_project_id(raw)
    except PathShapedProjectIdError as exc:
        raise _InvalidRequest('project_id', str(exc)) from exc
    err = validate_project_id(project_id) or validate_known_project_id(project_id, known_projects)
    if err:
        raise _InvalidRequest('project_id', err['error'])
    return project_id


def _resolvable_prefix(raw: object) -> str:
    if is_resolvable_prefix(raw):
        return raw
    raise _InvalidRequest('prefix', f'prefix must be {RESOLVABLE_PREFIX_GRAMMAR}')


def register_resolve_prefix_route(
    mcp: FastMCP,
    resolve: PrefixResolver,
    *,
    known_projects: dict[str, str] | None,
) -> None:
    """Mount POST /resolve-prefix on *mcp*'s HTTP app, answering through *resolve*.

    The request is validated in full before *resolve* is called, so a
    ValueError raised by a real bug inside the resolver is never reported as
    a client 400.
    """

    @mcp.custom_route(RESOLVE_PREFIX_PATH, methods=['POST'])
    async def resolve_prefix_route(request: Request) -> JSONResponse:
        try:
            body = _json_object(await request.body())
            project_id = _known_project_id(body.get('project_id'), known_projects)
            prefix = _resolvable_prefix(body.get('prefix'))
        except _InvalidRequest as exc:
            return JSONResponse(
                {'error': str(exc), 'error_type': 'ValidationError', 'field': exc.field},
                status_code=400,
            )
        try:
            resolution = await resolve(project_id, prefix)
        except ResolverUnavailable as exc:
            logger.warning(
                'POST %s: the %s store could not be read for project=%s prefix=%s',
                RESOLVE_PREFIX_PATH, exc.store, project_id, prefix,
                exc_info=True,
            )
            return JSONResponse(resolver_unavailable_body(exc), status_code=503)
        return JSONResponse(resolution_body(resolution))
