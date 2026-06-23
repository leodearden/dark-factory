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

    def test_submit_rejects_non_sentinel_agent_role(self, tmp_path: Path):
        """Non-sentinel --agent-role is rejected before any disk write.

        The CLI stamps level=2 directly, bypassing server.py's _chokepoint_or_submit
        downgrade gate.  A non-sentinel role (not harness-* or orchestrator-*) would
        produce a born-at-L2 record that violates the sentinel-namespace invariant, so
        submit.py rejects it via parser.error() (exit code 2) before any disk write.
        """
        with pytest.raises(SystemExit) as exc:
            submit.main([
                'submit',
                '--queue-dir', str(tmp_path),
                '--task', '5',
                '--severity', 'critical',
                '--category', 'infra_issue',
                '--summary', 's',
                '--agent-role', 'some-agent',  # not harness-* or orchestrator-*
            ])
        assert exc.value.code == 2, (
            f'Expected SystemExit(2) for non-sentinel agent_role, got {exc.value.code}'
        )
        # Nothing should have been written to the queue
        assert EscalationQueue(tmp_path).get_pending() == [], (
            'No escalation should be written when agent_role is rejected'
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
        Fails if the [project.scripts] table is missing, the 'escalation' key is absent,
        or the configured module:attr does not resolve to a callable.

        The exact string is intentionally NOT pinned: we verify the entry point
        resolves to a callable rather than asserting a literal, so benign moves
        (e.g. main() renamed or relocated) don't produce false failures.
        """
        import importlib
        import tomllib

        import escalation
        pyproject_path = Path(escalation.__file__).resolve().parents[2] / 'pyproject.toml'
        assert pyproject_path.exists(), f'pyproject.toml not found at {pyproject_path}'

        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)

        scripts = data.get('project', {}).get('scripts', {})
        ep_value = scripts.get('escalation', '')
        assert ep_value, (
            "Expected [project.scripts] to declare an 'escalation' console-script entry point"
        )
        # Verify the entry point is in 'module:attr' format and resolves to a callable.
        assert ':' in ep_value, (
            f"Entry point must be 'module:attr' format, got {ep_value!r}"
        )
        ep_module_name, ep_attr_name = ep_value.rsplit(':', 1)
        ep_mod = importlib.import_module(ep_module_name)
        ep_fn = getattr(ep_mod, ep_attr_name, None)
        assert callable(ep_fn), (
            f"Entry point {ep_value!r} must resolve to a callable, got {ep_fn!r}"
        )

    def test_console_script_form_if_installed(self, tmp_path: Path):
        """Behavioral test of the `escalation submit ...` console-script form.

        Exercises task ε's production seam: the systemd-run --on-failure target.
        Skipped when the console script is not installed in PATH (common in
        worktree dev environments before `uv sync --reinstall`).
        """
        import shutil
        escalation_script = shutil.which('escalation')
        if not escalation_script:
            pytest.skip('escalation console script not installed in PATH')

        proc = subprocess.run(
            [
                escalation_script,
                'submit',
                '--queue-dir', str(tmp_path),
                '--task', '99',
                '--severity', 'critical',
                '--category', 'infra_issue',
                '--summary', 'console-script behavioral test',
            ],
            capture_output=True,
        )
        assert proc.returncode == 0, (
            f'Expected returncode=0; stderr={proc.stderr.decode()!r}'
        )
        pending = EscalationQueue(tmp_path).get_pending()
        assert len(pending) == 1
        assert pending[0].level == 2
        assert pending[0].severity == 'critical'
