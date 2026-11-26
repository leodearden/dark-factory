"""Config-contract tests for fused-memory/fused-memory.service.example-systemd-config.

Guards the operational guarantee that MEM0_TELEMETRY=false is pinned at the
systemd/process layer — defense-in-depth so the telemetry-off guarantee does not
depend solely on Python import ordering in server/main.py.

Also guards WatchdogSec=120 in both committed unit files (task 1731 palliative):
ensures that even if the dedicated-thread heartbeat is delayed, systemd gives
the process 120s before SIGABRT (was 30s — too tight with the old on-loop design).
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from _loopback_host import is_loopback_host
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_systemd_unit(path: Path) -> dict[str, list[str]]:
    """Parse a systemd unit file into a dict of section → non-comment lines.

    Rules applied:
    - Lines whose stripped form starts with '[' open a new section.
    - Lines whose stripped form starts with '#' or ';' are comments — skip.
    - Blank lines are skipped.
    - All other lines belong to the current section (None before the first header).

    Limitation: line continuations (trailing '\\') are NOT joined.  Each
    physical line is recorded separately.  This is harmless for exact-string
    membership checks like ``Environment=MEM0_TELEMETRY=false``, but callers
    that need reassembled values for multi-line directives (e.g. ExecStart)
    must handle joining themselves.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSystemdUnitMem0Telemetry:
    """Guard that MEM0_TELEMETRY=false is present as a functional directive in [Service]."""

    # Locate unit file: tests/ → fused-memory/ (parent.parent) → file
    UNIT_FILE = (
        Path(__file__).resolve().parent.parent
        / "fused-memory.service.example-systemd-config"
    )
    EXPECTED_DIRECTIVE = "Environment=MEM0_TELEMETRY=false"

    def test_mem0_telemetry_false_in_service_section(self):
        """[Service] must contain Environment=MEM0_TELEMETRY=false as a non-comment directive.

        systemd only honours Environment= directives under [Service]; a directive
        placed in another section or hidden in a comment is silently ignored.
        A bare substring search anywhere in the file would produce a false-green
        on a misplaced or commented-out line — so we parse section-by-section.

        Background: mem0's Posthog telemetry historically spawned ~8000 threads/day
        and pegged CPU at 1400%. The in-process dotenv suppression in server/main.py
        is contingent on import ordering. Pinning the var at the systemd layer removes
        that fragile dependency (belt-and-suspenders; both mechanisms are kept active).
        The live/deployed unit must carry this identical line.

        Scope note: this test pins the *checked-in example* file
        (fused-memory.service.example-systemd-config), not the live/deployed unit.
        It cannot detect drift between the example and a deployed unit.  That gap
        is accepted as an inherent in-repo limitation; deployment discipline (copying
        the example to the actual unit) remains the responsibility of the operator.
        """
        sections = _parse_systemd_unit(self.UNIT_FILE)
        assert "Service" in sections, (
            f"[Service] section not found in {self.UNIT_FILE}"
        )
        service_directives = sections["Service"]
        assert self.EXPECTED_DIRECTIVE in service_directives, (
            f"'{self.EXPECTED_DIRECTIVE}' not found as a non-comment directive "
            f"inside the [Service] section of {self.UNIT_FILE.name}.\n"
            "Add:\n"
            "    Environment=MEM0_TELEMETRY=false\n"
            "under [Service] (alongside the existing Environment= lines) to pin "
            "mem0 Posthog telemetry OFF at the systemd/process layer. "
            "The live/deployed unit must carry the identical line.\n"
            f"Current [Service] directives: {service_directives}"
        )


# ---------------------------------------------------------------------------
# WatchdogSec contract — task 1731 palliative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit_path,label", [
    (
        Path(__file__).resolve().parent.parent / "fused-memory.service.example-systemd-config",
        "fused-memory.service.example-systemd-config",
    ),
    (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "fused-memory.service.template",
        "scripts/fused-memory.service.template",
    ),
])
class TestWatchdogSecContract:
    """Guard that WatchdogSec=120 is present as a non-comment [Service] directive
    in both committed unit files (task 1731 palliative: 30->120 to give the
    dedicated-thread watchdog heartbeat adequate headroom).

    Section-aware parse prevents a commented/misplaced line from false-greening.
    """

    EXPECTED_DIRECTIVE = "WatchdogSec=120"

    def test_watchdog_sec_120_in_service_section(self, unit_path: Path, label: str) -> None:
        """[Service] must contain WatchdogSec=120 as a non-comment directive.

        WatchdogSec=30 was too tight: the old on-loop coroutine heartbeat was
        silenced when the asyncio loop was busy/blocked >30s, causing systemd
        to SIGABRT the whole service ~6-8x/day (task 1731).  With the dedicated
        OS-thread heartbeat, 120s is belt-and-suspenders headroom rather than
        the primary guard.

        The live/deployed unit must be updated out-of-band by the operator
        (installed-vs-committed split; the worktree cannot commit ~/.config/...).
        """
        sections = _parse_systemd_unit(unit_path)
        assert "Service" in sections, (
            f"[Service] section not found in {label}"
        )
        service_directives = sections["Service"]
        assert self.EXPECTED_DIRECTIVE in service_directives, (
            f"'{self.EXPECTED_DIRECTIVE}' not found as a non-comment directive "
            f"inside the [Service] section of {label}.\n"
            "Change WatchdogSec=30 -> WatchdogSec=120 under [Service] "
            "(task 1731 palliative: gives the dedicated-thread heartbeat "
            "adequate headroom). The live/deployed unit must be updated "
            "out-of-band by the operator.\n"
            f"Current [Service] directives: {service_directives}"
        )


# ---------------------------------------------------------------------------
# PATH contract — task 4448
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit_path,label", [
    (
        Path(__file__).resolve().parent.parent / "fused-memory.service.example-systemd-config",
        "fused-memory.service.example-systemd-config",
    ),
    (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "fused-memory.service.template",
        "scripts/fused-memory.service.template",
    ),
])
class TestServicePathPinned:
    """Guard that Environment=PATH= is pinned in both committed unit files.

    THE INCIDENT (2026-08-13→08-18). The unit set no PATH, so it inherited
    whatever PATH the systemd user manager happened to hold at the time it was
    started. On an instance whose manager env lacked ``~/.local/bin``, the
    ``claude`` CLI did not resolve, every curator LLM call raised
    FileNotFoundError, and the fleet-wide task-dedupe function was down for
    >=80h with no signal — only a WARNING in a journal that rolls in ~5 days.

    Pinning PATH here is the belt to the braces of resolving argv[0] explicitly
    (shared.cli_invoke::resolve_claude_binary): with both in place, the failure
    needs the binary to be genuinely absent rather than merely unreachable.

    Section-aware parse prevents a commented/misplaced line from false-greening.
    """

    EXPECTED_PREFIX = "Environment=PATH="
    EXPECTED_ENTRY = "%h/.local/bin"

    def test_path_pinned_in_service_section(self, unit_path: Path, label: str) -> None:
        """[Service] must pin PATH, and that PATH must carry %h/.local/bin."""
        sections = _parse_systemd_unit(unit_path)
        assert "Service" in sections, f"[Service] section not found in {label}"
        service_directives = sections["Service"]
        path_lines = [d for d in service_directives if d.startswith(self.EXPECTED_PREFIX)]
        assert path_lines, (
            f"No '{self.EXPECTED_PREFIX}' directive found in the [Service] section "
            f"of {label}. Without it the unit inherits the systemd user manager's "
            "PATH, which is how the curator lost the `claude` binary for >=80h on "
            "2026-08-13→08-18.\n"
            f"Current [Service] directives: {service_directives}"
        )
        assert any(self.EXPECTED_ENTRY in line for line in path_lines), (
            f"The pinned PATH in {label} does not contain '{self.EXPECTED_ENTRY}', "
            "which is where the claude CLI installs. Use the %h specifier rather "
            "than a hardcoded home directory so the template stays host-portable "
            "and byte-identical to the rendered unit.\n"
            f"Found: {path_lines}"
        )


# ---------------------------------------------------------------------------
# Socket activation — the port survives service restarts
# ---------------------------------------------------------------------------

_FM_DIR = Path(__file__).resolve().parent.parent
_SOCKET_UNIT = _FM_DIR.parent / "scripts" / "fused-memory.socket.template"
_SERVICE_UNITS = [
    _FM_DIR / "fused-memory.service.example-systemd-config",
    _FM_DIR.parent / "scripts" / "fused-memory.service.template",
]


class TestSocketActivation:
    """fused-memory.socket must hold exactly the port the server is configured for,
    and every committed service unit must depend on it."""

    def test_socket_listens_on_the_configured_endpoints(self, monkeypatch) -> None:
        from fused_memory.config.schema import FusedMemoryConfig

        monkeypatch.setenv("CONFIG_PATH", str(_FM_DIR / "config" / "config.yaml"))
        monkeypatch.delenv("RECON_ESCALATION_PORT", raising=False)
        config = FusedMemoryConfig()
        recon = config.reconciliation
        assert _parse_systemd_unit(_SOCKET_UNIT)["Socket"] == [
            f"ListenStream={config.server.host}:{config.server.port}",
            f"ListenStream={recon.escalation_host}:{recon.escalation_port}",
        ]

    def test_socket_listens_on_loopback_only(self) -> None:
        listen_streams = [
            d for d in _parse_systemd_unit(_SOCKET_UNIT)["Socket"]
            if d.startswith("ListenStream=")
        ]
        assert listen_streams, f"{_SOCKET_UNIT.name} declares no ListenStream="
        for directive in listen_streams:
            host = directive.split("=", 1)[1].rpartition(":")[0]
            assert is_loopback_host(host), (
                f"{_SOCKET_UNIT.name}: {directive!r} does not bind a loopback IP address"
            )

    @pytest.mark.parametrize("unit_path", _SERVICE_UNITS, ids=lambda p: p.name)
    def test_service_wants_and_orders_after_the_socket(self, unit_path: Path) -> None:
        sections = _parse_systemd_unit(unit_path)
        unit = sections["Unit"]
        assert "Wants=fused-memory.socket" in unit
        after = [d for d in unit if d.startswith("After=")]
        assert any("fused-memory.socket" in d.split("=", 1)[1].split() for d in after)
        assert "Also=fused-memory.socket" in sections["Install"]

    @pytest.mark.parametrize("unit_path", _SERVICE_UNITS, ids=lambda p: p.name)
    def test_a_deliberate_stop_also_stops_the_socket(self, unit_path: Path) -> None:
        service = _parse_systemd_unit(unit_path)["Service"]
        hooks = [d for d in service if d.startswith("ExecStopPost=")]
        assert len(hooks) == 1
        assert hooks[0].endswith("/scripts/stop-socket-unless-restarting.sh %n")


@pytest.mark.parametrize("unit_path", _SERVICE_UNITS, ids=lambda p: p.name)
class TestKillModeMixed:
    """A stop must signal only the main process, so the server receives one
    SIGTERM and drains its in-flight tool calls (task 9493)."""

    def test_service_kill_mode_is_mixed(self, unit_path: Path) -> None:
        service = _parse_systemd_unit(unit_path)["Service"]
        kill_modes = [d for d in service if d.startswith("KillMode=")]
        assert kill_modes == ["KillMode=mixed"], (
            f"{unit_path.name} must declare exactly one KillMode=mixed under [Service]; "
            f"found {kill_modes}. With the default control-group, systemd SIGTERMs "
            "every process in the cgroup, so the server receives 2+ stop signals and "
            "the curator/judge subprocesses behind live tool calls are killed."
        )


# ---------------------------------------------------------------------------
# Local embedder start hooks and cutover hold (task 6797)
# ---------------------------------------------------------------------------

_SERVICE_TEMPLATE = _FM_DIR.parent / "scripts" / "fused-memory.service.template"
_HOLD_DROPIN = _FM_DIR.parent / "scripts" / "fused-memory-cutover-hold.conf"
_COMPOSE_FILE = _FM_DIR / "docker" / "docker-compose.yml"


def _exec_start_pre(unit_path: Path) -> list[str]:
    return [d for d in _parse_systemd_unit(unit_path)["Service"] if d.startswith("ExecStartPre=")]


def _command_argv(exec_line: str) -> list[str]:
    """The command an Exec*= line runs, without its directive name or `-` prefix."""
    return exec_line.split("=", 1)[1].removeprefix("-").split()


def _bound_secs(exec_line: str) -> int | None:
    """How long the line's command may run, or None when nothing bounds it."""
    argv = _command_argv(exec_line)
    if argv[0] == "/usr/bin/timeout":
        return int(argv[1])
    if argv[0].endswith("wait-for-port.py"):
        return int(argv[argv.index("--timeout") + 1])
    return None


def _required_bound_secs(exec_line: str) -> int:
    bound = _bound_secs(exec_line)
    assert bound is not None, exec_line
    return bound


def _published_embedder_address() -> str:
    """The host side of the embedder's compose publish, e.g. 127.0.0.1:8420."""
    compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
    host_ip, host_port, _container_port = compose["services"]["embedder"]["ports"][0].split(":")
    return f"{host_ip}:{host_port}"


_EMBEDDER_ADDRESS = _published_embedder_address()
_STORE_ADDRESSES = ("127.0.0.1:6379", "127.0.0.1:6333")


def _is_embedder_up(line: str) -> bool:
    argv = line.split()
    return "compose" in argv and "up" in argv and "embedder" in argv


def _is_embedder_wait(line: str) -> bool:
    return "wait-for-port" in line and _EMBEDDER_ADDRESS in line.split()


def _is_store_wait(line: str) -> bool:
    return "wait-for-port" in line and set(_STORE_ADDRESSES) <= set(line.split())


def _index_of(lines: list[str], is_wanted: Callable[[str], bool]) -> int:
    matches = [index for index, line in enumerate(lines) if is_wanted(line)]
    assert len(matches) == 1, matches
    return matches[0]


class TestEmbedderStartHooks:
    """The embedder hooks never block fused-memory's start and fit the start timeout."""

    def _embedder_lines(self) -> list[str]:
        return [
            line for line in _exec_start_pre(_SERVICE_TEMPLATE)
            if "embedder" in line or _EMBEDDER_ADDRESS in line
        ]

    def test_only_the_up_and_the_wait_concern_the_embedder_and_both_ignore_failure(self) -> None:
        embedder_lines = self._embedder_lines()
        assert len(embedder_lines) == 2
        assert sum(map(_is_embedder_up, embedder_lines)) == 1
        assert sum(map(_is_embedder_wait, embedder_lines)) == 1
        assert all(line.startswith("ExecStartPre=-") for line in embedder_lines)

    def test_every_embedder_hook_is_bounded(self) -> None:
        unbounded = [line for line in self._embedder_lines() if _bound_secs(line) is None]
        assert unbounded == []

    def test_embedder_up_never_pulls_never_starts_dependencies_and_names_only_the_embedder(self) -> None:
        lines = _exec_start_pre(_SERVICE_TEMPLATE)
        argv = lines[_index_of(lines, _is_embedder_up)].split()
        assert "--pull=never" in argv
        assert "--no-deps" in argv
        assert argv[-1] == "embedder"

    def test_embedder_is_started_before_the_store_wait_and_waited_on_after(self) -> None:
        lines = _exec_start_pre(_SERVICE_TEMPLATE)
        up = _index_of(lines, _is_embedder_up)
        store_wait = _index_of(lines, _is_store_wait)
        embedder_wait = _index_of(lines, _is_embedder_wait)
        assert up < store_wait < embedder_wait

    def test_store_wait_is_unprefixed_so_store_failure_still_fails_the_start(self) -> None:
        lines = _exec_start_pre(_SERVICE_TEMPLATE)
        assert lines[_index_of(lines, _is_store_wait)].startswith("ExecStartPre=/")

    def test_store_wait_and_embedder_bounds_fit_inside_the_start_timeout(self) -> None:
        service = _parse_systemd_unit(_SERVICE_TEMPLATE)["Service"]
        start_timeout = int(next(d for d in service if d.startswith("TimeoutStartSec=")).split("=", 1)[1])
        lines = _exec_start_pre(_SERVICE_TEMPLATE)
        store_wait = _required_bound_secs(lines[_index_of(lines, _is_store_wait)])
        embedder_bounds = [_required_bound_secs(line) for line in self._embedder_lines()]
        assert store_wait + sum(embedder_bounds) < start_timeout


class TestCutoverHoldDropin:
    """The drop-in holds both units while the lock exists, and the lock survives a reboot."""

    def _condition(self) -> str:
        unit = _parse_systemd_unit(_HOLD_DROPIN)
        assert list(unit) == ["Unit"]
        assert len(unit["Unit"]) == 1
        return unit["Unit"][0]

    def test_condition_is_negated_path_existence(self) -> None:
        assert self._condition().startswith("ConditionPathExists=!")

    def test_lock_path_is_reboot_surviving_and_host_portable(self) -> None:
        lock = self._condition().split("!", 1)[1]
        assert lock.startswith("%h/.local/state/")
        assert not lock.startswith(("/tmp", "/run", "/var/run"))
        assert "XDG_RUNTIME_DIR" not in lock
