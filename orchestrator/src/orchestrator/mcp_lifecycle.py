"""Start/stop fused-memory HTTP server as a managed subprocess."""

import asyncio
import logging

import httpx

from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class McpLifecycle:
    """Manages the fused-memory MCP HTTP server process."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config.fused_memory
        self.project_root = config.project_root
        self._process: asyncio.subprocess.Process | None = None

    @property
    def url(self) -> str:
        return self.config.url

    async def start(self) -> None:
        """Start fused-memory with HTTP transport, wait for health."""
        if self._process is not None:
            logger.warning('MCP server already running')
            return

        cmd = self.config.server_command
        config_path = (self.project_root / self.config.config_path).resolve()
        cmd = [*cmd, '--config', str(config_path)]

        logger.info(f'Starting fused-memory HTTP server: {" ".join(cmd)}')
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for health
        if not await self._wait_for_health(timeout=30):
            await self.stop()
            raise RuntimeError('Fused-memory server failed to start within 30s')

        logger.info(f'Fused-memory HTTP server ready at {self.config.url}')

    async def stop(self) -> None:
        """Graceful shutdown."""
        if self._process is None:
            return

        logger.info('Stopping fused-memory HTTP server')
        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except TimeoutError:
                logger.warning('Server did not stop gracefully, killing')
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            pass
        finally:
            self._process = None

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
            if self._process.returncode is not None:
                stderr = await self._process.stderr.read() if self._process.stderr else b''
                logger.error(f'Server exited prematurely: {stderr.decode()[-500:]}')
                return False
            if await self.health_check():
                return True
            await asyncio.sleep(0.5)
        return False

    def mcp_config_json(self) -> dict:
        """Return MCP server config dict suitable for --mcp-config."""
        return {
            'mcpServers': {
                'fused-memory': {
                    'type': 'streamable-http',
                    'url': f'{self.config.url}/mcp/',
                },
            },
        }
