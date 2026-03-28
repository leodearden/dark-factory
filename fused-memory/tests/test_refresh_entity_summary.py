"""Tests for refresh_entity_summary across backends, service, and MCP tool.

Covers:
- GraphitiBackend.get_valid_edges_for_node()
- GraphitiBackend.update_node_summary()
- GraphitiBackend.refresh_entity_summary()
- MemoryService.refresh_entity_summary()
- MCP tool refresh_entity_summary in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
- STAGE1_SYSTEM_PROMPT in prompts/stage1.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
