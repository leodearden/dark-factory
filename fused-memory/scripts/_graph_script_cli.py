"""How a read-only measurement script is pointed at FalkorDB, and how its output files land.

Shared by replay_overreach_extraction.py and score_overreach_composite.py.
Every graph read goes through
fused_memory/backends/falkor_read_only.py::ReadOnlyGraphReader. A script names
its graph by ``--graph-uri``, else env ``FALKORDB_URI``, else the config's
``graphiti.falkordb.uri``, or by ``--graph-host``/``--graph-port`` instead of a
uri. Its outputs are written only after every input has validated and every
output's directory exists, each then replaced atomically through
shared/src/shared/safe_io.py::atomic_write_text.

A script imports this module beside itself::

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _graph_script_cli import ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from redis.exceptions import RedisError
from shared.safe_io import atomic_write_text

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.config.schema import FusedMemoryConfig

logger = logging.getLogger('graph_script_cli')

ReaderFactory = Callable[[str], ReadOnlyGraphReader]
"""A graph name -> a reader on that graph."""

GRAPH_UNAVAILABLE: tuple[type[Exception], ...] = (OSError, RedisError)
"""What a read raises when FalkorDB is unreachable or refuses it; anything else is a defect."""


def add_graph_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--graph-uri', default=None,
                        help='FalkorDB URI. Default: env FALKORDB_URI, then config.')
    parser.add_argument('--graph-host', default=None, help='FalkorDB host, instead of a URI.')
    parser.add_argument('--graph-port', type=int, default=None, help='FalkorDB port, with --graph-host.')


def resolve_falkordb_uri(args: argparse.Namespace) -> str | None:
    """``--graph-uri``, then env ``FALKORDB_URI``, then config; None leaves the client default."""
    if args.graph_uri:
        return str(args.graph_uri)
    if env_uri := os.environ.get('FALKORDB_URI'):
        return env_uri
    try:
        return getattr(getattr(FusedMemoryConfig().graphiti, 'falkordb', None), 'uri', None)
    except Exception:
        logger.warning('no FalkorDB uri from config; using the client default', exc_info=True)
        return None


def live_reader_factory(args: argparse.Namespace) -> ReaderFactory:
    """Readers addressed by host/port when either is given, else by the resolved uri."""
    if args.graph_host is not None or args.graph_port is not None:
        return lambda graph: ReadOnlyGraphReader(graph_name=graph, host=args.graph_host, port=args.graph_port)
    uri = resolve_falkordb_uri(args)
    return lambda graph: ReadOnlyGraphReader(graph_name=graph, uri=uri)


class UnreadableInput(ValueError):
    """An input file that cannot be read or parsed."""


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnreadableInput(f'cannot read {path}: {exc}') from exc


def write_files_atomically(texts: Mapping[Path, str]) -> None:
    """Create every output's directory before any file is written, then replace each atomically."""
    for path in texts:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in texts.items():
        atomic_write_text(path, text)
