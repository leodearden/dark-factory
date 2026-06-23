"""Tests for escalation.submit CLI module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from escalation import submit
from escalation.queue import EscalationQueue


class TestSubmitCli:
    """In-process tests for submit.main() and subprocess `python -m escalation submit`."""

    def test_submit_main_writes_pending_l2(self, tmp_path: Path):
        """Happy path: submit.main() writes a born-at-L2 escalation to the queue.

        Verifies:
        - return code is 0
        - exactly one pending escalation exists in the queue
        - level == 2 (CLI stamps L2 directly since it bypasses the server)
        - task_id, severity, category, summary, detail are passed through
        - status == 'pending'
        - agent_role starts with 'orchestrator-' (default sentinel role)
        - severity is NOT downgraded (CLI bypasses server downgrade logic)
        """
        rc = submit.main([
            'submit',
            '--queue-dir', str(tmp_path),
            '--task', '4242',
            '--severity', 'critical',
            '--category', 'infra_issue',
            '--summary', 'self-restart OnFailure',
            '--detail', 'unit failed',
        ])

        assert rc == 0

        pending = EscalationQueue(tmp_path).get_pending()
        assert len(pending) == 1

        esc = pending[0]
        assert esc.level == 2, f'Expected level=2 (born-at-L2), got {esc.level}'
        assert esc.task_id == '4242'
        assert esc.severity == 'critical', 'Severity must NOT be downgraded by CLI'
        assert esc.category == 'infra_issue'
        assert esc.summary == 'self-restart OnFailure'
        assert esc.detail == 'unit failed'
        assert esc.status == 'pending'
        assert esc.agent_role.startswith('orchestrator-'), (
            f'Expected sentinel role starting with orchestrator-, got {esc.agent_role!r}'
        )

    def test_submit_rejects_non_l2_severity(self, tmp_path: Path):
        """Non-BORN_AT_L2_SEVERITIES severity is rejected before any disk write.

        The CLI exists to produce born-at-L2 records; a non-L2 severity (e.g.
        'blocking') would silently write an L0 record and defeat the purpose.
        Argparse choices= restriction gives exit code 2 before any file is created.
        """
        with pytest.raises(SystemExit) as exc:
            submit.main([
                'submit',
                '--queue-dir', str(tmp_path),
                '--task', '5',
                '--severity', 'blocking',
                '--category', 'infra_issue',
                '--summary', 's',
            ])

        assert exc.value.code == 2, (
            f'Expected SystemExit(2) from argparse choices rejection, got {exc.value.code}'
        )
        # Nothing should have been written to the queue
        assert EscalationQueue(tmp_path).get_pending() == [], (
            'No escalation should be written when severity is rejected'
        )

    def test_python_dash_m_escalation_submit_signal(self, tmp_path: Path):
        """User-observable signal: `python -m escalation submit ...` writes a born-at-L2 record.

        Uses a subprocess with PYTHONPATH pointing at the worktree src so the
        in-tree escalation package (with __main__.py and submit.py) wins over any
        editable install in the main checkout that might lack these new files.
        """
        import escalation
        src_dir = Path(escalation.__file__).resolve().parents[1]

        proc = subprocess.run(
            [
                sys.executable, '-m', 'escalation',
                'submit',
                '--queue-dir', str(tmp_path),
                '--task', '7',
                '--severity', 'urgent',
                '--category', 'infra_issue',
                '--summary', 's',
            ],
            env={**os.environ, 'PYTHONPATH': str(src_dir)},
            capture_output=True,
        )

        assert proc.returncode == 0, (
            f'Expected returncode=0; stderr={proc.stderr.decode()!r}'
        )

        pending = EscalationQueue(tmp_path).get_pending()
        assert len(pending) == 1
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'urgent'

    def test_pyproject_registers_escalation_console_entrypoint(self):
        """Seam-contract: escalation/pyproject.toml declares the console-script entrypoint.

        Protects task ε's systemd-run --on-failure target (`escalation submit ...`).
        Fails if the [project.scripts] table is missing or has a typo.
        """
        import importlib
        import tomllib

        import escalation
        pyproject_path = Path(escalation.__file__).resolve().parents[2] / 'pyproject.toml'
        assert pyproject_path.exists(), f'pyproject.toml not found at {pyproject_path}'

        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)

        scripts = data.get('project', {}).get('scripts', {})
        assert scripts.get('escalation') == 'escalation.submit:main', (
            f"Expected project.scripts.escalation = 'escalation.submit:main', got {scripts!r}"
        )

        # Also verify the target is importable and callable
        mod = importlib.import_module('escalation.submit')
        assert callable(mod.main), 'escalation.submit.main must be callable'
