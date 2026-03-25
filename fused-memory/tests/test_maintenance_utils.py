"""Tests for fused_memory.maintenance.utils._safe_close helper."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from fused_memory.maintenance.utils import _safe_close


class TestSafeClose:
    """Tests for _safe_close() async helper."""

    @pytest.mark.asyncio
    async def test_calls_service_close(self):
        """_safe_close awaits service.close() exactly once on the happy path."""
        service = AsyncMock()
        logger = logging.getLogger('test_calls_service_close')
        await _safe_close(service, logger, 'test_context')
        service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_raise_when_close_raises(self):
        """_safe_close does not propagate exceptions raised by service.close()."""
        service = AsyncMock()
        service.close = AsyncMock(side_effect=RuntimeError('boom'))
        logger = logging.getLogger('test_does_not_raise')
        # Must not raise
        await _safe_close(service, logger, 'test_context')

    @pytest.mark.asyncio
    async def test_logs_warning_with_context_name_on_close_error(self, caplog):
        """_safe_close logs a WARNING containing context_name when close() raises."""
        service = AsyncMock()
        service.close = AsyncMock(side_effect=RuntimeError('boom'))
        logger = logging.getLogger('test_logs_warning')
        with caplog.at_level(logging.WARNING, logger='test_logs_warning'):
            await _safe_close(service, logger, 'run_reindex')

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            'Error closing service during run_reindex cleanup' in m
            for m in warning_messages
        ), f'Expected warning with context name, got: {warning_messages}'

    @pytest.mark.asyncio
    async def test_logs_warning_with_exc_info(self, caplog):
        """_safe_close includes exc_info in the WARNING log record."""
        service = AsyncMock()
        service.close = AsyncMock(side_effect=ValueError('err'))
        logger = logging.getLogger('test_exc_info')
        with caplog.at_level(logging.WARNING, logger='test_exc_info'):
            await _safe_close(service, logger, 'run_cleanup')

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected at least one WARNING record'
        assert warning_records[0].exc_info is not None, (
            'Expected exc_info to be set on the WARNING record'
        )

    @pytest.mark.asyncio
    async def test_uses_provided_logger_not_module_logger(self, caplog):
        """_safe_close emits the warning under the caller-supplied logger, not the utils logger."""
        service = AsyncMock()
        service.close = AsyncMock(side_effect=RuntimeError('fail'))
        custom_logger = logging.getLogger('custom.test.logger')
        with caplog.at_level(logging.WARNING, logger='custom.test.logger'):
            await _safe_close(service, custom_logger, 'run_verify')

        # Warning should appear under 'custom.test.logger'
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'custom.test.logger'
        ]
        assert warning_records, (
            f'Expected WARNING under custom.test.logger, got records: {caplog.records}'
        )
        # Must NOT appear under the utils module logger
        utils_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'fused_memory.maintenance.utils'
        ]
        assert not utils_records, (
            f'Warning should not appear under utils module logger, got: {utils_records}'
        )

    @pytest.mark.asyncio
    async def test_noop_when_close_succeeds_no_warning(self, caplog):
        """_safe_close emits no WARNING records when service.close() succeeds."""
        service = AsyncMock()
        logger = logging.getLogger('test_noop')
        with caplog.at_level(logging.WARNING, logger='test_noop'):
            await _safe_close(service, logger, 'run_reindex')

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warning_records, f'Expected no warnings on success, got: {warning_records}'
