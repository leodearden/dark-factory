"""The cross-host drift probe, scripts/deploy/build_cache_drift_probe.sh.

It prints one `name<TAB>value` line per input that reaches a reify sccache key
(PRD C13, plans/canonical-build-path-shared-cache-prd.md). It runs here in a
controlled environment: PATH is a shim bin before /usr/bin:/bin, HOME is
scratch, and no compiler variable is set. The compilers, rustc, dpkg-query,
pkg-config and sccache are shims whose answers the tests change one at a
time. The host trees it digests (the OCCT headers, /opt/reify-deps and the
slvs header) are small scratch trees named by the probe's REIFY_PROBE_*
variables, so no run reads the real ones. The scratch reify checkout carries a
fixture Cargo.lock whose registry crates have sources under the scratch
$HOME/.cargo/registry/src: two with [package.metadata.system-deps] entries,
x11-dl with its build.rs probe list, one with neither, and one whose source is
not on the host.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from laptop_battery_guard_fixtures import write_shims
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

PROBE = Path(__file__).resolve().parents[1] / "deploy" / "build_cache_drift_probe.sh"
CLI_TIMEOUT = cli_timeout_from_env("BUILD_CACHE_DRIFT_PROBE_TEST_TIMEOUT")

REQUIRED_LINES = {
    "rustc.bin",
    "cc1",
    "cc1plus",
    "as",
    "dpkg.toolchain",
}

FIXTURE_MODULES = {"glib-2.0", "gobject-2.0", "gtk+-3.0", "gtk-3.0-fallback", "x11", "xscrnsaver", "zlib"}

_REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
_CARGO_LOCK = f"""\
version = 4

[[package]]
name = "glib-sys"
version = "0.18.1"
source = "{_REGISTRY}"

[[package]]
name = "gtk-sys"
version = "0.18.2"
source = "{_REGISTRY}"

[[package]]
name = "x11-dl"
version = "2.21.0"
source = "{_REGISTRY}"

[[package]]
name = "serde"
version = "1.0.200"
source = "{_REGISTRY}"

[[package]]
name = "winapi"
version = "0.3.9"
source = "{_REGISTRY}"

[[package]]
name = "reify-kernel"
version = "0.1.0"
"""

_CRATE_SOURCES = {
    "glib-sys-0.18.1": {
        "Cargo.toml": """\
[package]
name = "glib-sys"
version = "0.18.1"

[package.metadata.system-deps.glib_2_0]
name = "glib-2.0"
version = "2.56"

[package.metadata.system-deps.gobject_2_0]
name = "gobject-2.0"
version = "2.56"
""",
    },
    "gtk-sys-0.18.2": {
        "Cargo.toml": """\
[package]
name = "gtk-sys"
version = "0.18.2"

[package.metadata.system-deps]
zlib = "1.2"

[package.metadata.system-deps.gtk]
name = "gtk+-3.0"
fallback-names = ["gtk-3.0-fallback"]
version = "3.22"
""",
    },
    "x11-dl-2.21.0": {
        "Cargo.toml": '[package]\nname = "x11-dl"\nversion = "2.21.0"\n',
        "build.rs": """\
fn main() {
    let libraries = [
        // lib           pkgconfig name
        ("x11", "x11"),
        ("xscrnsaver", "xscrnsaver"),
    ];
}
""",
    },
    "serde-1.0.200": {"Cargo.toml": '[package]\nname = "serde"\nversion = "1.0.200"\n'},
}

_DPKG_QUERY = """\
import fnmatch, json, sys

packages = sorted(json.load(open(STATE))["packages"], key=lambda row: row["package"])
args = sys.argv[1:]
fmt = args[args.index("-f") + 1].replace("\\\\n", "\\n")
patterns = [arg for i, arg in enumerate(args) if not arg.startswith("-") and args[i - 1] != "-f"]
rc = 0
selected = []
for pattern in patterns or ["*"]:
    matches = [row for row in packages if fnmatch.fnmatchcase(row["package"], pattern)]
    if not matches:
        print(f"dpkg-query: no packages found matching {pattern}", file=sys.stderr)
        rc = 1
    selected.extend(row for row in matches if row not in selected)
for row in selected:
    sys.stdout.write(
        fmt.replace("${db:Status-Abbrev}", row["status"])
        .replace("${Package}", row["package"])
        .replace("${Version}", row["version"])
    )
sys.exit(rc)
"""

_PKG_CONFIG = """\
import json, sys

modules = json.load(open(STATE))["modules"]
args = sys.argv[1:]
if args == ["--list-all"]:
    for name in modules:
        print(f"{name:<24} {name} - fake module")
    sys.exit(0)
if len(args) == 2 and args[0] == "--variable=libdir":
    if args[1] in modules:
        print(f"/usr/lib/{args[1]}")
        sys.exit(0)
    sys.exit(1)
if args[:1] == ["--modversion"]:
    rc = 0
    for name in args[1:]:
        if name in modules:
            print(modules[name])
        else:
            print(f"Package {name} was not found in the pkg-config search path.", file=sys.stderr)
            rc = 1
    sys.exit(rc)
sys.exit(2)
"""


def write_python_shim(bin_dir: Path, name: str, state: Path, body: str) -> None:
    shim = bin_dir / name
    shim.write_text(f"#!{sys.executable} -I\nSTATE = {str(state)!r}\n{body}")
    shim.chmod(0o755)


class ProbeWorld:
    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.reify = tmp_path / "reify"
        (self.reify / "crates" / "reify-build-utils").mkdir(parents=True)
        (self.reify / "Cargo.lock").write_text(_CARGO_LOCK)
        index = self.home / ".cargo" / "registry" / "src" / "index.crates.io-1949cf8c6b5b557f"
        for crate, files in _CRATE_SOURCES.items():
            (index / crate).mkdir(parents=True)
            for name, text in files.items():
                (index / crate / name).write_text(text)
        (self.reify / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.96.0"\n')
        (self.reify / "crates" / "reify-build-utils" / "native-deps.toml").write_text(
            "[slvs]\n"
            'include_env = "SLVS_INCLUDE_DIR"\n'
            'lib_env = "SLVS_LIB_DIR"\n'
            f'include_candidates = ["{tmp_path}/deps/include"]\n'
            f'lib_candidates = ["{tmp_path}/deps/lib"]\n'
            'include_sentinel = "slvs.h"\n'
            'lib_sentinel = "libslvs.a"\n'
        )
        toolchain = tmp_path / "toolchain"
        self.sysroot = toolchain / "sysroot"
        rustlib = self.sysroot / "lib" / "rustlib" / "x86_64-unknown-linux-gnu" / "lib"
        rustlib.mkdir(parents=True)
        (self.sysroot / "lib" / "librustc_driver-fake.so").write_bytes(b"driver")
        (rustlib / "libstd-fake.rlib").write_bytes(b"std")
        self.cc1 = toolchain / "cc1"
        self.cc1.write_bytes(b"cc1 v1")
        self.cc1plus = toolchain / "cc1plus"
        self.cc1plus.write_bytes(b"cc1plus v1")

        self.bin = tmp_path / "bin"
        compiler = f"""\
case "$*" in
  "-dumpfullversion -dumpmachine") printf '13.3.0\\nx86_64-linux-gnu\\n' ;;
  -print-prog-name=cc1) echo {self.cc1} ;;
  -print-prog-name=cc1plus) echo {self.cc1plus} ;;
  *) exit 1 ;;
esac
"""
        write_shims(
            self.bin,
            {
                "rustup": "exit 1\n",
                "rustc": f"""\
case "$*" in
  -vV) printf '%s\\n' 'rustc 1.96.0 (ac68faa20 2026-10-01)' 'binary: rustc' \\
         'commit-hash: ac68faa20' 'host: x86_64-unknown-linux-gnu' \\
         'release: 1.96.0' 'LLVM version: 22.1.2' ;;
  "--print sysroot") echo {self.sysroot} ;;
  *) exit 1 ;;
esac
""",
                "cc": compiler,
                "c++": compiler,
                "as": "echo 'GNU assembler 2.42'\n",
                "sccache": "echo 'sccache 0.14.0'\n",
            },
        )
        self.dpkg_state = tmp_path / "dpkg.json"
        self.pkg_config_state = tmp_path / "pkg-config.json"
        self.set_packages(
            [
                {"status": "ii ", "package": "libc6-dev", "version": "2.39-0ubuntu8.6"},
                {"status": "ii ", "package": "libx11-dev", "version": "2:1.8.7-1build1"},
                {"status": "ii ", "package": "gcc-13", "version": "13.3.0-6ubuntu2"},
                {"status": "rc ", "package": "libold-dev", "version": "1.0"},
            ]
        )
        self.set_modules({"x11": "1.8.7", "zlib": "1.3", "glib-2.0": "2.80.0", "unrelated": "9.9"})
        write_python_shim(self.bin, "dpkg-query", self.dpkg_state, _DPKG_QUERY)
        write_python_shim(self.bin, "pkg-config", self.pkg_config_state, _PKG_CONFIG)

        host = tmp_path / "host"
        self.occt_include = host / "opencascade"
        self.occt_include.mkdir(parents=True)
        (self.occt_include / "Standard.hxx").write_text("// occt 7.8\n")
        (self.occt_include / "gp_Pnt.hxx").write_text("// point\n")
        self.deps_root = host / "reify-deps"
        for relative in ("lib", "manifold/lib", "conda-meta"):
            (self.deps_root / relative).mkdir(parents=True)
        (self.deps_root / "lib" / "libopenvdb.a").write_bytes(b"openvdb v1")
        (self.deps_root / "manifold" / "lib" / "libmanifold.a").write_bytes(b"manifold v1")
        (self.deps_root / "conda-meta" / "tbb-2022.0.json").write_text("{}")
        (self.deps_root / "conda-meta" / "history").write_text("")
        self.slvs_header = host / "slvs.h"
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "REIFY_PROBE_OCCT_INCLUDE": str(self.occt_include),
            "REIFY_PROBE_DEPS_ROOT": str(self.deps_root),
            "REIFY_PROBE_SLVS_HEADER": str(self.slvs_header),
        }

    def set_packages(self, packages: list[dict[str, str]]) -> None:
        self.dpkg_state.write_text(json.dumps({"packages": packages}))

    def packages(self) -> list[dict[str, str]]:
        return json.loads(self.dpkg_state.read_text())["packages"]

    def set_modules(self, modules: dict[str, str]) -> None:
        self.pkg_config_state.write_text(json.dumps({"modules": modules}))

    def modules(self) -> dict[str, str]:
        return json.loads(self.pkg_config_state.read_text())["modules"]

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(PROBE), *args],
            env={**self.env, **env},
            cwd=self.home,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
        )

    def probe(self) -> dict[str, str]:
        proc = self.run(str(self.reify))
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.splitlines()
        assert all("\t" in line for line in lines), lines
        pairs = [line.split("\t", 1) for line in lines]
        names = [name for name, _ in pairs]
        assert len(names) == len(set(names)), names
        return dict(pairs)


def changed_lines(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {
        name
        for name in before.keys() | after.keys()
        if not name.startswith("#info") and before.get(name) != after.get(name)
    }


def tree_snapshot(root: Path) -> dict[str, Any]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


@pytest.fixture
def world(tmp_path: Path) -> ProbeWorld:
    return ProbeWorld(tmp_path)


class TestDriftProbe:
    def test_prints_unique_named_lines_including_the_key_inputs(
        self, world: ProbeWorld
    ) -> None:
        lines = world.probe()
        assert set(lines) >= REQUIRED_LINES
        assert all(name for name in lines)

    def test_two_runs_are_identical(self, world: ProbeWorld) -> None:
        assert world.probe() == world.probe()

    def test_cc1_bytes_change_only_the_cc1_line(self, world: ProbeWorld) -> None:
        before = world.probe()
        world.cc1.write_bytes(b"cc1 v2")
        assert changed_lines(before, world.probe()) == {"cc1"}

    def test_prints_one_pkg_config_line_per_module_the_lock_probes(
        self, world: ProbeWorld
    ) -> None:
        lines = world.probe()
        derived = {name.removeprefix("pkg-config.") for name in lines if name.startswith("pkg-config.")}
        assert derived == FIXTURE_MODULES | {"slvs"}
        assert lines["pkg-config.glib-2.0"] == "2.80.0; libdir=/usr/lib/glib-2.0"
        assert lines["pkg-config.xscrnsaver"] == "ABSENT"
        assert lines["#info.cargo.registry-sources-missing"] == "winapi-0.3.9"

    def test_a_fetched_source_without_system_deps_changes_no_compared_line(
        self, world: ProbeWorld
    ) -> None:
        before = world.probe()
        winapi = world.home / ".cargo" / "registry" / "src" / "index.crates.io-1949cf8c6b5b557f" / "winapi-0.3.9"
        winapi.mkdir()
        (winapi / "Cargo.toml").write_text('[package]\nname = "winapi"\nversion = "0.3.9"\n')
        after = world.probe()
        assert changed_lines(before, after) == set()
        assert after["#info.cargo.registry-sources-missing"] == ""

    def test_a_module_going_present_changes_only_its_line(self, world: ProbeWorld) -> None:
        before = world.probe()
        world.set_modules({**world.modules(), "xscrnsaver": "1.2.3"})
        after = world.probe()
        assert changed_lines(before, after) == {"pkg-config.xscrnsaver"}
        assert after["pkg-config.xscrnsaver"] == "1.2.3; libdir=/usr/lib/xscrnsaver"

    def test_a_module_going_absent_changes_only_its_line(self, world: ProbeWorld) -> None:
        before = world.probe()
        world.set_modules({name: v for name, v in world.modules().items() if name != "zlib"})
        assert changed_lines(before, world.probe()) == {"pkg-config.zlib"}

    def test_an_unprobed_module_or_dev_package_changes_nothing(
        self, world: ProbeWorld
    ) -> None:
        before = world.probe()
        world.set_modules({**world.modules(), "libfoo": "1.0"})
        world.set_packages(
            [*world.packages(), {"status": "ii ", "package": "libxss-dev", "version": "1:1.2.3-1build3"}]
        )
        assert changed_lines(before, world.probe()) == set()

    @pytest.mark.parametrize(
        ("change", "line"),
        [
            ("occt-header", "occt.include.tree"),
            ("static-archive", "reify-deps.static-archives"),
            ("conda-package", "reify-deps.conda-meta"),
            ("slvs-header", "pkg-config.slvs"),
        ],
    )
    def test_a_host_tree_change_changes_only_its_line(
        self, world: ProbeWorld, change: str, line: str
    ) -> None:
        before = world.probe()
        if change == "occt-header":
            (world.occt_include / "Standard.hxx").write_text("// occt 7.9\n")
        elif change == "static-archive":
            (world.deps_root / "manifold" / "lib" / "libmanifold.a").write_bytes(b"manifold v2")
        elif change == "conda-package":
            (world.deps_root / "conda-meta" / "zstd-1.5.json").write_text("{}")
        else:
            world.slvs_header.write_text("/* slvs */\n")
        assert changed_lines(before, world.probe()) == {line}

    def test_writes_nothing(self, world: ProbeWorld) -> None:
        reify_before = tree_snapshot(world.reify)
        home_before = tree_snapshot(world.home)
        world.probe()
        assert tree_snapshot(world.reify) == reify_before
        assert tree_snapshot(world.home) == home_before

    def test_a_checkout_without_a_cargo_lock_exits_two(self, world: ProbeWorld) -> None:
        (world.reify / "Cargo.lock").unlink()
        proc = world.run(str(world.reify))
        assert proc.returncode == 2
        assert proc.stdout == ""
        assert "Cargo.lock" in proc.stderr

    def test_a_missing_reify_checkout_exits_two(self, world: ProbeWorld, tmp_path: Path) -> None:
        proc = world.run(str(tmp_path / "no-such-reify"))
        assert proc.returncode == 2
        assert proc.stdout == ""
