"""Unit tests for fused_memory.maintenance.seed_autopilot_video_triage_guardrails.

All tests import the target module inside the test body so that pytest can
collect this file even before the implementation exists (the collection step
itself will not raise ModuleNotFoundError; only test execution will fail).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def guardrails_source(tmp_path: Path) -> Path:
    """Write a minimal autopilot-video guardrails.py stub and return its path.

    The stub mirrors the real source's public contract: the two constants
    with all routing-critical keys (category, project_id, dual_write,
    metadata.{type,target,name}), plus a get_guardrail_payloads helper
    that shallow-copies metadata (Task 416 regression guard). Content
    strings are intentionally short — the loader tests assert on
    structure, not prose.
    """
    stub = tmp_path / "guardrails.py"
    stub.write_text(
        '"""Fixture stub of autopilot-video guardrails.py for test isolation."""\n'
        'from __future__ import annotations\n'
        '\n'
        'ATTRIBUTION_STUB_GUARDRAIL = {\n'
        '    "content": "stub content for attribution guardrail",\n'
        '    "category": "preferences_and_norms",\n'
        '    "project_id": "autopilot_video",\n'
        '    "dual_write": True,\n'
        '    "metadata": {\n'
        '        "type": "behavioral_guardrail",\n'
        '        "target": "triage_agent",\n'
        '        "name": "attribution_stub_anti_pattern",\n'
        '    },\n'
        '}\n'
        '\n'
        'NAV_HINTS_GUARDRAIL = {\n'
        '    "content": "stub content for nav hints guardrail",\n'
        '    "category": "preferences_and_norms",\n'
        '    "project_id": "autopilot_video",\n'
        '    "dual_write": True,\n'
        '    "metadata": {\n'
        '        "type": "behavioral_guardrail",\n'
        '        "target": "triage_agent",\n'
        '        "name": "nav_hints_anti_pattern",\n'
        '    },\n'
        '}\n'
        '\n'
        '\n'
        'def get_guardrail_payloads(agent_id):\n'
        '    payloads = []\n'
        '    for g in (ATTRIBUTION_STUB_GUARDRAIL, NAV_HINTS_GUARDRAIL):\n'
        '        payload = {**g, "metadata": dict(g["metadata"]), "agent_id": agent_id}\n'
        '        payloads.append(payload)\n'
        '    return payloads\n'
    )
    return stub


# ---------------------------------------------------------------------------
# loader tests
# ---------------------------------------------------------------------------


def test_load_guardrail_payloads_returns_two_dicts(guardrails_source: Path):
    """load_guardrail_payloads returns a list of exactly two dicts."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    payloads = load_guardrail_payloads(agent_id="test-agent", source_path=guardrails_source)

    assert isinstance(payloads, list)
    assert len(payloads) == 2
    for p in payloads:
        assert isinstance(p, dict)


def test_load_guardrail_payloads_routing_values(guardrails_source: Path):
    """Each payload must have routing-critical values for Mem0 add_memory."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    payloads = load_guardrail_payloads(agent_id="test-agent", source_path=guardrails_source)

    for p in payloads:
        assert isinstance(p["content"], str) and p["content"], (
            f"Expected non-empty str for content, got {p['content']!r}"
        )
        assert p["category"] == "preferences_and_norms", (
            f"Expected 'preferences_and_norms', got {p['category']!r}"
        )
        assert p["project_id"] == "autopilot_video", (
            f"Expected 'autopilot_video', got {p['project_id']!r}"
        )
        assert p["dual_write"] is True, (
            f"Expected dual_write=True, got {p['dual_write']!r}"
        )
        assert p["metadata"]["type"] == "behavioral_guardrail", (
            f"Expected type='behavioral_guardrail', got {p['metadata']['type']!r}"
        )


def test_load_guardrail_payloads_metadata_names(guardrails_source: Path):
    """The metadata name set must be exactly the two expected guardrail names."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    payloads = load_guardrail_payloads(agent_id="test-agent", source_path=guardrails_source)

    names = {p["metadata"]["name"] for p in payloads}
    assert names == {"attribution_stub_anti_pattern", "nav_hints_anti_pattern"}, (
        f"Unexpected names: {names!r}"
    )


def test_load_guardrail_payloads_agent_id_injected(guardrails_source: Path):
    """agent_id is injected into every returned payload."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    payloads = load_guardrail_payloads(
        agent_id="claude-task-1040-implementer", source_path=guardrails_source
    )

    for p in payloads:
        assert p["agent_id"] == "claude-task-1040-implementer", (
            f"Expected 'claude-task-1040-implementer', got {p['agent_id']!r}"
        )


def test_load_guardrail_payloads_missing_source_raises():
    """load_guardrail_payloads raises FileNotFoundError for a nonexistent source_path."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    missing = Path("/tmp/does-not-exist-1040.py")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_guardrail_payloads(agent_id="x", source_path=missing)

    assert str(missing) in str(exc_info.value), (
        f"Expected path {missing!s} in error message, got: {exc_info.value}"
    )


def test_load_guardrail_payloads_syntax_error_wraps_with_path(tmp_path: Path):
    """A SyntaxError in the source file surfaces with the source path named."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    broken = tmp_path / "guardrails.py"
    broken.write_text("def def def oops\n")

    with pytest.raises(RuntimeError) as exc_info:
        load_guardrail_payloads(agent_id="x", source_path=broken)

    assert str(broken) in str(exc_info.value), (
        f"Expected path {broken!s} in error, got: {exc_info.value}"
    )
    assert "SyntaxError" in str(exc_info.value) or isinstance(
        exc_info.value.__cause__, SyntaxError
    )


def test_load_guardrail_payloads_missing_symbol_wraps_with_path(tmp_path: Path):
    """A module missing get_guardrail_payloads surfaces a pointed error."""
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    stub = tmp_path / "guardrails.py"
    stub.write_text('"""Valid module with no get_guardrail_payloads."""\n')

    with pytest.raises(RuntimeError) as exc_info:
        load_guardrail_payloads(agent_id="x", source_path=stub)

    msg = str(exc_info.value)
    assert str(stub) in msg, f"Expected path {stub!s} in error, got: {msg}"
    assert "get_guardrail_payloads" in msg, (
        f"Expected symbol name in error, got: {msg}"
    )


def test_load_guardrail_payloads_metadata_not_aliased_across_calls(guardrails_source: Path):
    """Two successive calls return independent metadata dict objects.

    The loader deep-copies its output, so mutations to call-A's metadata
    dicts must not alias into call-B's — regardless of whether the source
    module's own get_guardrail_payloads copies metadata internally.  This
    tests the loader's own non-aliasing guarantee (amendment 3), not the
    upstream autopilot-video module's behavior.
    """
    from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
        load_guardrail_payloads,
    )

    payloads_a = load_guardrail_payloads(agent_id="agent-a", source_path=guardrails_source)
    payloads_b = load_guardrail_payloads(agent_id="agent-b", source_path=guardrails_source)

    # Mutate call-A's metadata dicts in place
    for p in payloads_a:
        p["metadata"]["injected_by_test"] = "call-a-mutation"

    # Call-B's metadata must be unaffected
    for p in payloads_b:
        assert "injected_by_test" not in p["metadata"], (
            f"Call-B metadata was aliased to call-A: {p['metadata']!r}"
        )


# ---------------------------------------------------------------------------
# SeedManager tests
# ---------------------------------------------------------------------------


class TestSeedManager:
    @pytest.mark.asyncio
    async def test_seed_calls_add_memory_per_payload(self, guardrails_source: Path):
        """SeedManager.seed() calls service.add_memory once per payload."""
        from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
            SeedManager,
            load_guardrail_payloads,
        )

        mock_service = MagicMock()
        mock_service.add_memory = AsyncMock(
            return_value=MagicMock(memory_ids=["mem-1"], stores_written=[])
        )

        manager = SeedManager(mock_service)
        await manager.seed(agent_id="agent-X", source_path=guardrails_source)

        assert mock_service.add_memory.await_count == 2

        # Compare each call's kwargs against the expected payloads
        expected_payloads = load_guardrail_payloads("agent-X", source_path=guardrails_source)
        actual_calls = mock_service.add_memory.call_args_list
        assert len(actual_calls) == 2

        # Build a name-keyed dict from actual calls for order-independent comparison
        actual_by_name: dict[str, dict] = {}
        for call in actual_calls:
            kwargs = call.kwargs
            actual_by_name[kwargs["metadata"]["name"]] = kwargs

        for expected in expected_payloads:
            name = expected["metadata"]["name"]
            assert name in actual_by_name, f"No call found for guardrail {name!r}"
            actual_kwargs = actual_by_name[name]
            for key, val in expected.items():
                assert actual_kwargs.get(key) == val, (
                    f"Mismatch for guardrail {name!r} key {key!r}: "
                    f"expected {val!r}, got {actual_kwargs.get(key)!r}"
                )

    @pytest.mark.asyncio
    async def test_seed_returns_report_with_memory_ids(self, guardrails_source: Path):
        """seed() returns a SeedReport whose memory_ids_by_name has exactly two entries."""
        from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
            SeedManager,
            SeedReport,
        )

        mock_service = MagicMock()
        mock_service.add_memory = AsyncMock(
            return_value=MagicMock(memory_ids=["mem-1"], stores_written=[])
        )

        manager = SeedManager(mock_service)
        report = await manager.seed(agent_id="agent-X", source_path=guardrails_source)

        assert isinstance(report, SeedReport)
        assert set(report.memory_ids_by_name.keys()) == {
            "attribution_stub_anti_pattern",
            "nav_hints_anti_pattern",
        }
        for name, ids in report.memory_ids_by_name.items():
            assert isinstance(ids, list) and ids, (
                f"Expected non-empty list for {name!r}, got {ids!r}"
            )

    @pytest.mark.asyncio
    async def test_seed_raises_on_empty_memory_ids(self, guardrails_source: Path):
        """seed() raises PartialSeedError naming the failing guardrail when memory_ids=[].

        The raised PartialSeedError must carry a partial_report attribute so an
        operator can tell which guardrails already succeeded (and therefore which
        Graphiti episodes may need cleanup before a retry).
        """
        from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
            PartialSeedError,
            SeedManager,
            SeedReport,
        )

        mock_service = MagicMock()
        # First call returns empty memory_ids; second call would succeed
        mock_service.add_memory = AsyncMock(
            side_effect=[
                MagicMock(memory_ids=[], stores_written=[]),
                MagicMock(memory_ids=["ok"], stores_written=[]),
            ]
        )

        manager = SeedManager(mock_service)

        with pytest.raises(RuntimeError) as exc_info:
            await manager.seed(agent_id="agent-X", source_path=guardrails_source)

        error_msg = str(exc_info.value)
        assert "attribution_stub_anti_pattern" in error_msg, (
            f"Expected guardrail name in error, got: {error_msg!r}"
        )
        assert "empty memory_ids" in error_msg, (
            f"Expected 'empty memory_ids' in error, got: {error_msg!r}"
        )

        # The exception must be a PartialSeedError with a partial_report so
        # operators know which writes completed before the failure.
        assert isinstance(exc_info.value, PartialSeedError), (
            f"Expected PartialSeedError, got {type(exc_info.value).__name__}"
        )
        partial = exc_info.value.partial_report
        assert isinstance(partial, SeedReport), (
            f"Expected SeedReport, got {type(partial).__name__}"
        )
        # The first guardrail write failed — no successful writes in the partial report.
        assert partial.memory_ids_by_name == {}, (
            f"Expected empty memory_ids_by_name on failure of first write, "
            f"got {partial.memory_ids_by_name!r}"
        )

    @pytest.mark.asyncio
    async def test_seed_does_not_proceed_after_empty_memory_ids(self, guardrails_source: Path):
        """seed() fast-fails after the first empty memory_ids — does not call add_memory a second time."""
        from fused_memory.maintenance.seed_autopilot_video_triage_guardrails import (
            SeedManager,
        )

        mock_service = MagicMock()
        mock_service.add_memory = AsyncMock(
            side_effect=[
                MagicMock(memory_ids=[], stores_written=[]),
                MagicMock(memory_ids=["ok"], stores_written=[]),
            ]
        )

        manager = SeedManager(mock_service)

        with pytest.raises(RuntimeError):
            await manager.seed(agent_id="agent-X", source_path=guardrails_source)

        assert mock_service.add_memory.await_count == 1, (
            f"Expected exactly 1 add_memory call after fast-fail, "
            f"got {mock_service.add_memory.await_count}"
        )
