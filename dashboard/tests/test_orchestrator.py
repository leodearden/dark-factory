"""Tests for dashboard.data.orchestrator — orchestrator discovery and status."""

from __future__ import annotations


class TestFindRunningOrchestrators:
    """Tests for find_running_orchestrators — scans ps aux for orchestrator processes."""

    def test_parses_orchestrator_lines(self):
        """Two orchestrator lines with --prd flags produce two dicts with pid, prd, running, started."""
        from unittest.mock import patch
        import subprocess

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1234  0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator --prd /home/leo/prd1.md\n"
            "leo       5678  0.3  0.8 234567  4567 ?        Sl   10:30   0:02 python -m orchestrator --prd /home/leo/prd2.md\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 2
        assert result[0]['pid'] == 1234
        assert result[0]['prd'] == '/home/leo/prd1.md'
        assert result[0]['running'] is True
        assert isinstance(result[0]['started'], str)
        assert result[1]['pid'] == 5678
        assert result[1]['prd'] == '/home/leo/prd2.md'

    def test_filters_out_grep_process(self):
        """A 'grep orchestrator' line in ps output is excluded from results."""
        from unittest.mock import patch
        import subprocess

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1234  0.5  1.2 123456  7890 ?        Sl   Mar18   0:05 python -m orchestrator --prd /home/leo/prd1.md\n"
            "leo       9999  0.0  0.0  12345   678 pts/0    S+   10:31   0:00 grep orchestrator\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert len(result) == 1
        assert result[0]['pid'] == 1234

    def test_no_orchestrators_running(self):
        """No orchestrator lines in ps output returns empty list."""
        from unittest.mock import patch
        import subprocess

        from dashboard.data.orchestrator import find_running_orchestrators

        ps_output = (
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
            "leo       1111  0.1  0.5  54321  1234 ?        Ss   Mar17   1:23 /usr/bin/bash\n"
        )
        mock_result = subprocess.CompletedProcess(args=['ps', 'aux'], returncode=0, stdout=ps_output, stderr='')

        with patch('dashboard.data.orchestrator.subprocess.run', return_value=mock_result):
            result = find_running_orchestrators()

        assert result == []

    def test_subprocess_failure(self):
        """subprocess.run raising an exception returns empty list."""
        from unittest.mock import patch

        from dashboard.data.orchestrator import find_running_orchestrators

        with patch('dashboard.data.orchestrator.subprocess.run', side_effect=OSError('ps not found')):
            result = find_running_orchestrators()

        assert result == []
