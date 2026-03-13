"""Entry point for the Fused Memory MCP server."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logging.getLogger('mcp.server.streamable_http_manager').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def configure_uvicorn_logging():
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uv_logger.addHandler(handler)
        uv_logger.propagate = False


async def run_server():
    """Parse args, load config, init service, run MCP transport."""
    parser = argparse.ArgumentParser(description='Fused Memory MCP Server')
    default_config = Path(__file__).resolve().parents[3] / 'config' / 'config.yaml'
    parser.add_argument(
        '--config', type=Path, default=default_config,
        help='Path to YAML configuration file',
    )
    parser.add_argument(
        '--transport', choices=['stdio', 'sse', 'http'],
        help='Transport override',
    )
    args = parser.parse_args()

    # Set CONFIG_PATH for the settings source
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    if args.transport:
        config.server.transport = args.transport

    logger.info('Fused Memory MCP Server starting')
    logger.info(f'  LLM: {config.llm.provider}/{config.llm.model}')
    logger.info(f'  Embedder: {config.embedder.provider}/{config.embedder.model}')
    logger.info(f'  Graphiti: {config.graphiti.provider} ({config.graphiti.falkordb.uri})')
    logger.info(f'  Mem0/Qdrant: {config.mem0.qdrant_url}')
    logger.info(f'  Transport: {config.server.transport}')

    # Initialize service
    memory_service = MemoryService(config)
    await memory_service.initialize()

    # Create MCP server
    mcp = create_mcp_server(memory_service)
    mcp.settings.host = config.server.host
    mcp.settings.port = config.server.port

    # Run transport
    transport = config.server.transport
    logger.info(f'Starting MCP server with transport: {transport}')

    if transport == 'stdio':
        await mcp.run_stdio_async()
    elif transport == 'sse':
        await mcp.run_sse_async()
    elif transport == 'http':
        display_host = 'localhost' if config.server.host == '0.0.0.0' else config.server.host
        logger.info(f'  MCP Endpoint: http://{display_host}:{config.server.port}/mcp/')
        configure_uvicorn_logging()
        await mcp.run_streamable_http_async()
    else:
        raise ValueError(f'Unsupported transport: {transport}')


def main():
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise


if __name__ == '__main__':
    main()
