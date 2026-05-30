"""Tests for the target-verify-subprocess venv-isolation scrub (verify.py).

Guards the 2026-05-29 ghost-venv fix: a target project's verify/build/test
subprocess must NOT inherit the orchestrator's own venv/uv activation vars, or
the target's ``uv`` resolves dark-factory/.venv and a target ``uv sync``
corrupts the orchestrator runtime interpreter. See
orchestrator/src/orchestrator/verify.py: ``_VENV_ISOLATION_KEYS`` /
``_strip_venv_bin_from_path`` / ``_target_subprocess_env``.
"""

import os

from orchestrator import verify
from orchestrator.verify import (
    _strip_venv_bin_from_path,
    _target_subprocess_env,
)

ORCH_VENV = "/home/leo/src/dark-factory/.venv"


class TestTargetSubprocessEnvScrub:
    def test_venv_isolation_keys_removed_and_path_stripped(self, monkeypatch):
        fake_environ = {
            "VIRTUAL_ENV": ORCH_VENV,
            "UV_PROJECT_ENVIRONMENT": "/x/env",
            "UV_PROJECT": "/home/leo/src/dark-factory/orchestrator",
            "UV_ACTIVE": "1",
            "UV_FROZEN": "1",
            "UV_NO_SYNC": "1",
            "UV_RUN_RECURSION_DEPTH": "1",
            "CONDA_PREFIX": "/opt/conda",
            "CONDA_DEFAULT_ENV": "base",
            "PYTHONHOME": "/opt/python",
            # uv run prepends the venv bin dir; it must be stripped.
            "PATH": f"{ORCH_VENV}/bin:/home/leo/.cargo/bin:/usr/bin:/bin",
            "HOME": "/home/leo",
            "LANG": "en_US.UTF-8",
        }
        monkeypatch.setattr(verify.os, "environ", fake_environ)

        env = _target_subprocess_env(None)

        # every venv/uv/conda activation var is gone
        for key in (
            "VIRTUAL_ENV",
            "UV_PROJECT_ENVIRONMENT",
            "UV_PROJECT",
            "UV_ACTIVE",
            "UV_FROZEN",
            "UV_NO_SYNC",
            "UV_RUN_RECURSION_DEPTH",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "PYTHONHOME",
        ):
            assert key not in env, f"{key} leaked into target subprocess env"

        # the leading venv-bin is removed; the rest of PATH survives, in order
        path_parts = env["PATH"].split(os.pathsep)
        assert f"{ORCH_VENV}/bin" not in path_parts
        assert path_parts == ["/home/leo/.cargo/bin", "/usr/bin", "/bin"]

        # non-venv vars pass through; PYTHONUNBUFFERED is set unconditionally
        assert env["HOME"] == "/home/leo"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_overlay_survives_scrub_and_wins(self, monkeypatch):
        # reify's Rust verify env (the overlay from _resolve_verify_env) must
        # survive the scrub AND override os.environ — denylist removes only the
        # python-env-selection vars, never the cargo/sccache/jobserver vars.
        fake_environ = {
            "VIRTUAL_ENV": ORCH_VENV,
            "PATH": f"{ORCH_VENV}/bin:/home/leo/.cargo/bin:/usr/bin",
            "CARGO_INCREMENTAL": "1",  # overlay should override this
        }
        monkeypatch.setattr(verify.os, "environ", fake_environ)

        overlay = {
            "RUSTC_WRAPPER": "sccache",
            "CARGO_INCREMENTAL": "0",
            "CARGO_MAKEFLAGS": "--jobserver-auth=fifo:/tmp/reify-jobserver",
            "DF_VERIFY_ROLE": "merge",
        }
        env = _target_subprocess_env(overlay)

        assert env["RUSTC_WRAPPER"] == "sccache"
        assert env["CARGO_INCREMENTAL"] == "0"  # overlay wins over os.environ
        assert env["CARGO_MAKEFLAGS"].startswith("--jobserver-auth=fifo:")
        assert env["DF_VERIFY_ROLE"] == "merge"  # role preserved (task + merge)
        assert "VIRTUAL_ENV" not in env
        assert f"{ORCH_VENV}/bin" not in env["PATH"].split(os.pathsep)
        assert "/home/leo/.cargo/bin" in env["PATH"].split(os.pathsep)

    def test_noop_when_no_virtual_env_active(self, monkeypatch):
        # If the orchestrator is launched without a venv (no VIRTUAL_ENV), the
        # PATH strip is a no-op and nothing is removed.
        fake_environ = {"PATH": "/home/leo/.cargo/bin:/usr/bin", "HOME": "/home/leo"}
        monkeypatch.setattr(verify.os, "environ", fake_environ)
        env = _target_subprocess_env(None)
        assert env["PATH"] == "/home/leo/.cargo/bin:/usr/bin"
        assert env["PYTHONUNBUFFERED"] == "1"


class TestStripVenvBinFromPath:
    def test_removes_leading_venv_bin(self):
        path = f"{ORCH_VENV}/bin:/home/leo/.cargo/bin:/usr/bin"
        assert _strip_venv_bin_from_path(path, ORCH_VENV) == (
            "/home/leo/.cargo/bin:/usr/bin"
        )

    def test_removes_venv_bin_even_when_not_leading(self):
        path = f"/usr/bin:{ORCH_VENV}/bin:/bin"
        assert _strip_venv_bin_from_path(path, ORCH_VENV) == "/usr/bin:/bin"

    def test_noop_when_venv_is_none(self):
        assert _strip_venv_bin_from_path("/a:/b", None) == "/a:/b"

    def test_noop_when_path_is_none(self):
        assert _strip_venv_bin_from_path(None, ORCH_VENV) is None

    def test_trailing_slash_on_venv_is_normalized(self):
        path = f"{ORCH_VENV}/bin:/usr/bin"
        assert _strip_venv_bin_from_path(path, ORCH_VENV + "/") == "/usr/bin"
