"""Start/stop fused-memory HTTP server as a managed subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from shared.proc_group import terminate_process_group

from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)

# Retry settings for transient MCP failures (e.g. server restarting)
_RETRYABLE_STATUS = frozenset({502, 503, 504})
_MCP_MAX_RETRIES = 3
_MCP_BACKOFF_BASE = 1.0  # seconds; exponential: 1s, 2s, 4s

MCP_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream',
}


class McpSession:
    """Manages a single MCP Streamable HTTP session with initialization handshake."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.mcp_endpoint = f'{self.base_url}/mcp'
        self._session_id: str | None = None
        self._initialized = False
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def initialize(self) -> None:
        """Perform MCP initialize + initialized handshake."""
        if self._initialized:
            return

        # Step 1: initialize request (no session ID)
        result = await self._raw_call(
            'initialize',
            {
                'protocolVersion': '2025-03-26',
                'capabilities': {},
                'clientInfo': {'name': 'orchestrator', 'version': '0.1.0'},
            },
        )
        logger.debug(f'MCP initialize response: {json.dumps(result)[:200]}')

        # Step 2: Send initialized notification
        await self._raw_notify('notifications/initialized')

        self._initialized = True
        logger.info(f'MCP session initialized (session_id={self._session_id})')

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30,
    ) -> dict:
        """Call an MCP tool and return the result."""
        if not self._initialized:
            await self.initialize()

        result = await self._raw_call(
            'tools/call',
            {'name': name, 'arguments': arguments},
            timeout=timeout,
        )
        return result

    async def _raw_call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> dict:
        """Send a JSON-RPC request with retry on transient failures."""
        request_id = self._next_id()
        payload: dict[str, Any] = {
            'jsonrpc': '2.0',
            'id': request_id,
            'method': method,
        }
        if params is not None:
            payload['params'] = params

        last_exc: Exception | None = None
        for attempt in range(_MCP_MAX_RETRIES):
            headers = dict(MCP_HEADERS)
            if self._session_id:
                headers['Mcp-Session-Id'] = self._session_id

            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    resp = await client.post(
                        self.mcp_endpoint,
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )
                    if resp.status_code in _RETRYABLE_STATUS:
                        logger.warning(
                            'MCP %s returned %d (attempt %d/%d)',
                            method, resp.status_code, attempt + 1, _MCP_MAX_RETRIES,
                        )
                        self._session_id = None
                        self._initialized = False
                        last_exc = httpx.HTTPStatusError(
                            f'{resp.status_code}', request=resp.request, response=resp,
                        )
                        await asyncio.sleep(_MCP_BACKOFF_BASE * (2 ** attempt))
                        continue

                    resp.raise_for_status()

                    # Track session ID
                    resp_session_id = resp.headers.get('mcp-session-id')
                    if resp_session_id:
                        self._session_id = resp_session_id

                    return self._parse_response(resp)

            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                logger.warning(
                    'MCP %s connection error (attempt %d/%d): %s',
                    method, attempt + 1, _MCP_MAX_RETRIES, exc,
                )
                self._session_id = None
                self._initialized = False
                last_exc = exc
                await asyncio.sleep(_MCP_BACKOFF_BASE * (2 ** attempt))

        raise last_exc or RuntimeError('_raw_call exhausted retries')

    async def _raw_notify(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification with retry on transient failures."""
        payload: dict[str, Any] = {
            'jsonrpc': '2.0',
            'method': method,
        }
        if params is not None:
            payload['params'] = params

        last_exc: Exception | None = None
        for attempt in range(_MCP_MAX_RETRIES):
            headers = dict(MCP_HEADERS)
            if self._session_id:
                headers['Mcp-Session-Id'] = self._session_id

            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    resp = await client.post(
                        self.mcp_endpoint,
                        json=payload,
                        headers=headers,
                        timeout=10,
                    )
                    # Notifications may return 200 or 202
                    if resp.status_code not in (200, 202, 204):
                        logger.warning(
                            f'Notification {method} returned {resp.status_code}: {resp.text[:200]}'
                        )
                    return

            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                logger.warning(
                    'MCP notify %s connection error (attempt %d/%d): %s',
                    method, attempt + 1, _MCP_MAX_RETRIES, exc,
                )
                self._session_id = None
                self._initialized = False
                last_exc = exc
                await asyncio.sleep(_MCP_BACKOFF_BASE * (2 ** attempt))

        raise last_exc or RuntimeError('_raw_notify exhausted retries')

    @staticmethod
    def _parse_response(resp: httpx.Response) -> dict:
        """Parse JSON or SSE response."""
        content_type = resp.headers.get('content-type', '')

        if 'text/event-stream' in content_type:
            return _parse_sse_response(resp.text)
        elif 'application/json' in content_type:
            return resp.json()
        else:
            try:
                return resp.json()
            except (json.JSONDecodeError, ValueError):
                return _parse_sse_response(resp.text)


def _parse_sse_response(text: str) -> dict:
    """Parse SSE text to extract the JSON-RPC result."""
    last_data = None
    for line in text.split('\n'):
        if line.startswith('data: '):
            last_data = line[6:]
        elif line.startswith('data:'):
            last_data = line[5:]
    if last_data:
        return json.loads(last_data)
    raise ValueError(f'No data line found in SSE response: {text[:200]}')


# Module-level session singleton (created by McpLifecycle after server starts)
_session: McpSession | None = None


def get_session() -> McpSession:
    """Get the current MCP session. Raises if not initialized."""
    if _session is None:
        raise RuntimeError('MCP session not initialized — call McpLifecycle.start() first')
    return _session


async def mcp_call(
    url: str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 30,
    session_id: str | None = None,
) -> dict:
    """Send a JSON-RPC request to an MCP Streamable HTTP endpoint.

    Uses the module-level session if available, otherwise creates a one-shot session.
    """
    global _session

    if _session is not None:
        # Use existing session — extract tool name and args from params
        if method == 'tools/call' and params:
            return await _session.call_tool(
                params['name'], params.get('arguments', {}), timeout=timeout
            )
        else:
            return await _session._raw_call(method, params, timeout=timeout)

    # Fallback: create a one-shot session
    session = McpSession(url.rstrip('/mcp').rstrip('/'))
    await session.initialize()
    if method == 'tools/call' and params:
        return await session.call_tool(
            params['name'], params.get('arguments', {}), timeout=timeout
        )
    return await session._raw_call(method, params, timeout=timeout)


class McpLifecycle:
    """Manages the fused-memory MCP HTTP server process."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config.fused_memory
        self.project_root = config.project_root
        self._process: asyncio.subprocess.Process | None = None
        # pgid captured at spawn (pgid == pid under start_new_session); kept
        # alongside _process so stop() never has to call os.getpgid, which
        # would be unsafe if the PID has been reused post-reap.
        self._pgid: int | None = None

    @property
    def url(self) -> str:
        return self.config.url

    async def start(self) -> None:
        """Start fused-memory with HTTP transport, wait for health."""
        global _session

        if self._process is not None:
            logger.warning('MCP server already running')
            return

        if not self.config.server_command:
            # External server mode — skip subprocess, just connect
            logger.info(f'Connecting to external fused-memory at {self.config.url}')
            if not await self._wait_for_health(timeout=30):
                raise RuntimeError(
                    f'External fused-memory at {self.config.url} not reachable within 30s'
                )
            logger.info(f'Fused-memory HTTP server ready at {self.config.url}')
        else:
            cmd = self.config.server_command
            config_path = (self.project_root / self.config.config_path).resolve()
            cmd = [*cmd, '--config', str(config_path)]

            logger.info(f'Starting fused-memory HTTP server: {" ".join(cmd)}')
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._pgid = self._process.pid

            # Wait for health
            if not await self._wait_for_health(timeout=30):
                await self.stop()
                raise RuntimeError('Fused-memory server failed to start within 30s')

            logger.info(f'Fused-memory HTTP server ready at {self.config.url}')

        # Initialize MCP session
        _session = McpSession(self.config.url)
        await _session.initialize()

    async def stop(self) -> None:
        """Graceful shutdown."""
        global _session

        if self._process is None:
            return

        logger.info('Stopping fused-memory HTTP server')
        _session = None
        try:
            if self._pgid is not None:
                await terminate_process_group(self._process, self._pgid, grace_secs=10.0)
        except Exception:
            pass
        finally:
            self._process = None
            self._pgid = None

    async def health_check(self) -> bool:
        """Check if the server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f'{self.config.url}/health', timeout=5)
                return resp.status_code == 200
        except Exception:
            return False

    async def _wait_for_health(self, timeout: float = 30) -> bool:
        """Poll health endpoint until ready or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                stderr = await self._process.stderr.read() if self._process.stderr else b''
                logger.error(f'Server exited prematurely: {stderr.decode()[-500:]}')
                return False
            if await self.health_check():
                return True
            await asyncio.sleep(0.5)
        return False

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        """Return MCP server config dict suitable for --mcp-config."""
        config = {
            'mcpServers': {
                'fused-memory': {
                    'type': 'http',
                    'url': f'{self.config.url}/mcp',
                },
                'jcodemunch': {
                    'command': 'uvx',
                    'args': ['jcodemunch-mcp'],
                },
            },
        }
        if escalation_url:
            config['mcpServers']['escalation'] = {
                'type': 'http',
                'url': escalation_url,
            }
        return config
