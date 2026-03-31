"""Tests for merge_entities across backends, service, and MCP tool.

Covers:
- GraphitiBackend.redirect_node_edges()
- GraphitiBackend.delete_entity_node()
- GraphitiBackend.merge_entities()
- MemoryService.merge_entities()
- MCP tool merge_entities in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError
