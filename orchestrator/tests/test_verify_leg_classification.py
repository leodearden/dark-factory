"""Per-leg classification: what verify knows about a leg beyond its output.

``verify_classify.classify_failure`` answers what a tool's OUTPUT says. Two
facts it cannot see are known only to verify itself (task 4843):

- which SLOT the leg filled (lint, type or test), which names the gate that
  went red when the output names no more specific cause;
- that verify interposed ``/bin/bash`` around every leg, so a positive
  ``128 + N`` exit status is that shell reporting its child's death by
  signal N.

This module pins both, through the public predicate and the leg classifier.
"""

from __future__ import annotations

import pytest

from orchestrator.verify_classify import (
    is_external_kill_rc,
    is_shell_reported_external_kill,
)


class TestShellReportedExternalKill:
    """``is_shell_reported_external_kill(rc)`` reads a shell's ``128 + N``
    status for the external-kill signals, minus SIGKILL."""

    @pytest.mark.parametrize(
        ('rc', 'signal_name'),
        [(143, 'SIGTERM'), (130, 'SIGINT'), (129, 'SIGHUP')],
    )
    def test_external_kill_signals_are_recognised(self, rc, signal_name):
        assert is_shell_reported_external_kill(rc) is True, (
            f'rc={rc} is a shell reporting its child died of {signal_name}'
        )

    def test_sigkill_is_excluded(self):
        """Positive 137 is the ordinary branch-caused OOM of a grandchild,
        the residual the INFRA_KILL row already documents, so it must stay a
        code-fault exit status."""
        assert is_shell_reported_external_kill(137) is False

    @pytest.mark.parametrize('rc', [139, 134, 135, 132, 136])
    def test_crash_signals_are_code_faults(self, rc):
        assert is_shell_reported_external_kill(rc) is False, (
            f'rc={rc} is a crash signal, a fault of the code under test'
        )

    @pytest.mark.parametrize('rc', [0, 1, 2, 4, 5, 124, 127, 128, 255])
    def test_ordinary_exit_statuses_are_not_kills(self, rc):
        assert is_shell_reported_external_kill(rc) is False

    @pytest.mark.parametrize('rc', [-15, -9])
    def test_raw_negative_returncodes_belong_to_is_external_kill_rc(self, rc):
        assert is_shell_reported_external_kill(rc) is False

    def test_is_external_kill_rc_is_not_widened(self):
        """Task #5132: the bare predicate keeps refusing shell conventions;
        only the caller that interposed the shell may read them."""
        assert is_external_kill_rc(143) is False
