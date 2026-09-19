"""Tests for sampler.metrics — pure PSI parser and process-metric counters.

All functions under test take injected data (fixture text, fake process objects,
fd9-exists predicates) so they are fully deterministic and safe to run in pytest.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Realistic /proc/pressure/* text with both some and full lines
PSI_CPU_TEXT = (
    'some avg10=2.50 avg60=1.80 avg300=1.20 total=123456\n'
    'full avg10=0.30 avg60=0.20 avg300=0.10 total=9876\n'
)

# Memory pressure — only 'some' line present (full missing)
PSI_MEM_TEXT = 'some avg10=1.23 avg60=0.90 avg300=0.50 total=654321\n'

# IO pressure — both lines
PSI_IO_TEXT = (
    'some avg10=0.75 avg60=0.60 avg300=0.40 total=11111\n'
    'full avg10=0.45 avg60=0.30 avg300=0.20 total=2222\n'
)


# ---------------------------------------------------------------------------
# cgroup-tree fixtures (PRD plans/load-throttle-harmonisation-prd.md §6.4/§7)
#
# Every own-pressure test drives discovery and reading through a tmp_path tree
# plus a matching ``0::`` kernel path, so nothing here reads the live
# /sys/fs/cgroup — whose topology differs per host and per day.
# ---------------------------------------------------------------------------

# The segment prefix a systemd user manager puts every user unit under. The
# ``user@<uid>.service`` element is the discovery ANCHOR (see
# sampler.metrics.discover_pressure_cgroups).
USER_MANAGER_PREFIX = 'user.slice/user-1000.slice/user@1000.service'

# Parent slices of the two topologies PRD §6.4 names. ``df.slice`` is the
# post-3394 shape (PRD §7 row 1); ``app.slice`` is the shape measured live on
# this host on 2026-09-13, where 3394 has not landed and there is no df-*.slice
# anywhere — so the orchestrator-*.service fallback is today's LIVE branch.
DF_PARENT = 'df.slice'
APP_PARENT = 'app.slice'

# The seven leaves measured under app.slice on this host, and the 0:: path the
# sampler's own process reports there.
LIVE_ORCHESTRATOR_LEAVES = (
    'orchestrator-autopilot-video.service',
    'orchestrator-dark-factory.service',
    'orchestrator-know-live.service',
    'orchestrator-my-solar-challenge.service',
    'orchestrator-pump-web-ui.service',
    'orchestrator-reify.service',
    'orchestrator-solar-challenge-platform.service',
)
LIVE_OWN_CGROUP_PATH = f'/{USER_MANAGER_PREFIX}/{APP_PARENT}/orchestrator-dark-factory.service'

# A cpu.pressure body that no parser can extract an avg10 from — the "exists
# but is garbage" case, which must degrade exactly like an absent file.
GARBAGE_PRESSURE_TEXT = 'nothing here resembles a pressure line\n'


def pressure_text(*, some: float, full: float | None = 0.0) -> str:
    """Return a cpu.pressure body in the shape measured on a live cgroup.

    ``full=None`` omits the ``full`` line entirely — a legitimate kernel state
    for CPU pressure, which shared.psi.parse_pressure_file reads as a partial
    miss (a dict, not None).
    """
    lines = [f'some avg10={some:.2f} avg60=0.10 avg300=0.07 total=717833958']
    if full is not None:
        lines.append(f'full avg10={full:.2f} avg60=0.05 avg300=0.03 total=350000000')
    return '\n'.join(lines) + '\n'


class CgroupTree(NamedTuple):
    """The seams a built fixture tree exposes.

    ``cgroup_root`` and ``own_cgroup_path`` are what
    sampler.metrics.discover_pressure_cgroups takes; ``proc_cgroup_path`` is
    the α seam (shared.psi.resolve_own_cgroup) that produces the same
    ``own_cgroup_path`` from a ``0::`` line, so a test can prove the fixture's
    kernel-path text and its directory layout agree.
    """

    cgroup_root: Path
    own_cgroup_path: str
    proc_cgroup_path: Path


def build_cgroup_tree(
    root: Path,
    *,
    groups: Mapping[str, Mapping[str, str | None]],
    outside: Mapping[str, str] = MappingProxyType({}),
    own_cgroup_path: str = LIVE_OWN_CGROUP_PATH,
) -> CgroupTree:
    """Write a sysfs-shaped cgroup tree under *root* and return its seams.

    Args:
        root: A tmp_path. The tree is written under ``root/cgroup``.
        groups: ``{parent_slice_name: {leaf_name: cpu_pressure_text_or_None}}``,
            planted under the ``user@1000.service`` anchor. A ``None`` text
            creates the leaf directory with NO cpu.pressure file.
        outside: ``{leaf_name: text}`` planted under ``root/cgroup/system.slice``
            — i.e. OUTSIDE the anchor, so an anchored search must not find them.
        own_cgroup_path: The ``0::`` kernel path the sampler's own process
            reports. Also written to ``root/proc_self_cgroup``.
    """
    cgroup_root = root / 'cgroup'
    anchor = cgroup_root / USER_MANAGER_PREFIX
    for parent, leaves in groups.items():
        for leaf, text in leaves.items():
            leaf_dir = anchor / parent / leaf
            leaf_dir.mkdir(parents=True, exist_ok=True)
            if text is not None:
                (leaf_dir / 'cpu.pressure').write_text(text)
    for leaf, text in outside.items():
        leaf_dir = cgroup_root / 'system.slice' / leaf
        leaf_dir.mkdir(parents=True, exist_ok=True)
        (leaf_dir / 'cpu.pressure').write_text(text)

    proc_cgroup_path = root / 'proc_self_cgroup'
    proc_cgroup_path.write_text(f'0::{own_cgroup_path}\n')
    return CgroupTree(cgroup_root, own_cgroup_path, proc_cgroup_path)


def live_topology(root: Path, **pressures: str | None) -> CgroupTree:
    """Build topology (a): the seven app.slice leaves, no df-*.slice anywhere.

    Keyword overrides are keyed by leaf name with '.' and '-' replaced by '_',
    so a test can make one leaf's cpu.pressure absent or garbage without
    restating the other six.
    """
    leaves = {
        name: pressures.get(
            name.replace('.', '_').replace('-', '_'),
            pressure_text(some=round(0.5 + i, 2)),
        )
        for i, name in enumerate(LIVE_ORCHESTRATOR_LEAVES)
    }
    return build_cgroup_tree(root, groups={APP_PARENT: leaves})


def df_topology(root: Path, *project_ids: str) -> CgroupTree:
    """Build topology (b): one ``df-<project_id>.slice`` leaf per id, post-3394."""
    leaves = {
        f'df-{pid}.slice': pressure_text(some=round(1.5 + i, 2))
        for i, pid in enumerate(project_ids)
    }
    return build_cgroup_tree(root, groups={DF_PARENT: leaves})



# ---------------------------------------------------------------------------
# Step-1 tests: parse_pressure_file and collect_psi
# ---------------------------------------------------------------------------


class TestParsePressureFile:
    """The re-exported shared parser (see ``TestParserRehomedToShared``).

    These exact-dict pins therefore move whenever ``shared.psi`` grows a key:
    the four-key shape below is the 60 s window ruling D17 added (task 3353).
    ``collect_psi``'s own six-key contract is unaffected — it selects the two
    avg10 keys explicitly — which is what makes the growth additive here.
    """

    def test_both_lines_extracted(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_CPU_TEXT)
        assert result == {
            'some_avg10': 2.50,
            'some_avg60': 1.80,
            'full_avg10': 0.30,
            'full_avg60': 0.20,
        }

    def test_missing_full_defaults_to_zero(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_MEM_TEXT)
        assert result == {
            'some_avg10': 1.23,
            'some_avg60': 0.90,
            'full_avg10': 0.0,
            'full_avg60': 0.0,
        }

    def test_io_both_lines(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_IO_TEXT)
        assert result == {
            'some_avg10': 0.75,
            'some_avg60': 0.60,
            'full_avg10': 0.45,
            'full_avg60': 0.30,
        }

    def test_float_precision(self):
        from sampler.metrics import parse_pressure_file

        text = 'some avg10=99.99 avg60=0.00 avg300=0.00 total=0\n'
        result = parse_pressure_file(text)
        assert result is not None
        assert result['some_avg10'] == pytest.approx(99.99)
        assert result['full_avg10'] == 0.0

    def test_total_parse_miss_returns_none(self):
        """A total parse miss (no some/full avg10 line) must return None sentinel.

        Current code pre-seeds {some_avg10:0.0, full_avg10:0.0} and always
        returns that dict — these assertions FAIL (RED) until step-4 implements
        the sentinel.  The partial-miss boundary pin (PSI_MEM_TEXT has only
        'some') is included to verify None is NOT triggered for partial misses.
        """
        from sampler.metrics import parse_pressure_file

        # Total miss — garbage text with no recognisable avg10 fields
        assert parse_pressure_file('garbage line with no avg fields\n') is None
        # Empty string — also a total miss
        assert parse_pressure_file('') is None
        # Partial miss (some present, full absent) — NOT a total miss, must not be None
        assert parse_pressure_file(PSI_MEM_TEXT) is not None


class TestCollectPsi:
    def _fake_read(self, mapping: dict[str, str]):
        """Return a closure that looks up fixture text by name."""
        def read(name: str) -> str:
            return mapping[name]
        return read

    def test_returns_exactly_six_keys(self):
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        result = collect_psi(read=self._fake_read(mapping))
        expected_keys = {
            'psi_cpu_some_avg10', 'psi_cpu_full_avg10',
            'psi_mem_some_avg10', 'psi_mem_full_avg10',
            'psi_io_some_avg10', 'psi_io_full_avg10',
        }
        assert set(result.keys()) == expected_keys

    def test_values_match_parsed_text(self):
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        result = collect_psi(read=self._fake_read(mapping))

        assert result['psi_cpu_some_avg10'] == pytest.approx(2.50)
        assert result['psi_cpu_full_avg10'] == pytest.approx(0.30)
        assert result['psi_mem_some_avg10'] == pytest.approx(1.23)
        assert result['psi_mem_full_avg10'] == 0.0          # missing full -> 0.0
        assert result['psi_io_some_avg10'] == pytest.approx(0.75)
        assert result['psi_io_full_avg10'] == pytest.approx(0.45)

    def test_all_values_are_floats(self):
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        result = collect_psi(read=self._fake_read(mapping))
        for key, val in result.items():
            assert isinstance(val, float), f'{key} is not float: {val!r}'

    def test_total_miss_skips_source_and_warns(self, caplog):
        """collect_psi must skip a source whose parse returns None and emit WARNING.

        Current code (before step-6) passes parsed['some_avg10'] etc. directly,
        which would raise TypeError after step-4 changes parse to return None on
        a total miss.  Against step-4 code this test FAILS with TypeError (not
        the skip+warn behaviour) — confirming RED either way.
        """
        from sampler.metrics import collect_psi

        # cpu gets garbage — total miss -> None from parse_pressure_file
        # memory and io get valid text — should still appear in the result
        mapping = {
            'cpu': 'garbage no avg10\n',
            'memory': PSI_MEM_TEXT,
            'io': PSI_IO_TEXT,
        }
        with caplog.at_level(logging.WARNING):
            result = collect_psi(read=self._fake_read(mapping))

        # cpu keys must be absent — no fabricated 0.0
        assert 'psi_cpu_some_avg10' not in result
        assert 'psi_cpu_full_avg10' not in result
        # memory and io keys must be present
        assert set(result) == {
            'psi_mem_some_avg10', 'psi_mem_full_avg10',
            'psi_io_some_avg10', 'psi_io_full_avg10',
        }
        # At least one WARNING record must mention the skipped source name
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('cpu' in msg for msg in warning_messages), (
            f'Expected a WARNING mentioning "cpu"; got: {warning_messages}'
        )


# ---------------------------------------------------------------------------
# Step-3 tests: count_occt_queue_depth, count_verify_concurrency, sum_verify_rss
# ---------------------------------------------------------------------------


class FakeProc:
    """Lightweight process stub exposing the attributes the impl reads."""

    def __init__(
        self,
        pid: int,
        name: str = 'bash',
        cmdline: list[str] | None = None,
        rss: int = 0,
        children: list[FakeProc] | None = None,
    ):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline if cmdline is not None else []
        self._rss = rss
        self._children = children if children is not None else []

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return self._cmdline

    def memory_info(self):
        class _MI:
            def __init__(self, rss):
                self.rss = rss
        return _MI(self._rss)

    def children(self, recursive: bool = False) -> list[FakeProc]:
        if recursive:
            result = []
            for ch in self._children:
                result.append(ch)
                result.extend(ch.children(recursive=True))
            return result
        return list(self._children)


class TestCountOcctQueueDepth:
    def test_counts_only_occt_processes_waiting_on_semaphore(self):
        """Processes matching cargo-test-occt-gated bash without fd/9 are counted."""
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(1, 'bash', ['bash', 'cargo-test-occt-gated']),   # waiting (fd9 absent)
            FakeProc(2, 'bash', ['bash', 'cargo-test-occt-gated']),   # has fd9 → not waiting
            FakeProc(3, 'bash', ['bash', 'cargo-test-occt-gated']),   # waiting
            FakeProc(4, 'python', ['python', 'something']),            # not occt
        ]
        fd9_has = {2}  # pid 2 has fd/9

        def fd9_exists(pid: int) -> bool:
            return pid in fd9_has

        result = count_occt_queue_depth(procs, fd9_exists)
        assert result == 2

    def test_non_matching_processes_ignored(self):
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(10, 'python', ['python', 'script.py']),
            FakeProc(11, 'bash', ['bash', 'other-script']),
        ]
        result = count_occt_queue_depth(procs, lambda pid: False)
        assert result == 0

    def test_all_waiting(self):
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(i, 'bash', ['bash', 'cargo-test-occt-gated']) for i in range(5)
        ]
        result = count_occt_queue_depth(procs, lambda pid: False)
        assert result == 5

    def test_all_have_fd9_so_none_counted(self):
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(i, 'bash', ['bash', 'cargo-test-occt-gated']) for i in range(3)
        ]
        result = count_occt_queue_depth(procs, lambda pid: True)
        assert result == 0

    def test_realistic_path_with_extension_counted(self):
        """The real script is invoked with its .sh path — must be recognised.

        Realistic invocation: ['/bin/bash', 'reify/scripts/cargo-test-occt-gated.sh']
        The bare-token form ['bash', 'cargo-test-occt-gated'] was the only fixture
        used in prior tests, giving false confidence.  This test proves the
        path-with-extension form is also counted.

        Current _is_occt_gated uses list membership ('cargo-test-occt-gated' in cmdline),
        which requires an element to equal the bare token exactly.
        'reify/scripts/cargo-test-occt-gated.sh' != 'cargo-test-occt-gated', so count==0
        and this test FAILS against the current impl (confirming the RED phase).
        """
        from sampler.metrics import count_occt_queue_depth

        procs = [
            # Realistic form: bash running the script by its full relative path
            FakeProc(1, 'bash', ['/bin/bash', 'reify/scripts/cargo-test-occt-gated.sh']),
            # Another common form: bash + path with leading ./
            FakeProc(2, 'bash', ['bash', './scripts/cargo-test-occt-gated.sh', 'cargo', 'test']),
            # Unrelated process — must NOT be counted
            FakeProc(3, 'python', ['python', 'unrelated']),
        ]
        result = count_occt_queue_depth(procs, lambda _pid: False)
        # pids 1 and 2 match (path contains 'cargo-test-occt-gated'), pid 3 does not
        assert result == 2


class TestCountVerifyConcurrency:
    def test_counts_verify_sh_argv0(self):
        from sampler.metrics import count_verify_concurrency

        procs = [
            FakeProc(1, 'bash', ['verify.sh', 'arg1']),              # matches
            FakeProc(2, 'bash', ['/usr/local/bin/verify.sh', 'x']),  # basename matches
            FakeProc(3, 'bash', ['bash', 'verify.sh']),               # verify.sh NOT argv[0]
            FakeProc(4, 'python', ['python', 'app.py']),
        ]
        result = count_verify_concurrency(procs)
        assert result == 2

    def test_no_verify_processes(self):
        from sampler.metrics import count_verify_concurrency

        procs = [FakeProc(1, 'python', ['python', 'main.py'])]
        result = count_verify_concurrency(procs)
        assert result == 0

    def test_empty_cmdline_skipped(self):
        from sampler.metrics import count_verify_concurrency

        procs = [FakeProc(1, 'bash', [])]
        result = count_verify_concurrency(procs)
        assert result == 0


class TestSumVerifyRss:
    def test_sums_process_and_children_rss(self):
        from sampler.metrics import sum_verify_rss

        child1 = FakeProc(101, 'bash', [], rss=512)
        child2 = FakeProc(102, 'bash', [], rss=256)
        parent = FakeProc(100, 'bash', ['verify.sh'], rss=1024, children=[child1, child2])
        other = FakeProc(200, 'python', ['python'], rss=9999)

        result = sum_verify_rss([parent, other])
        # parent (1024) + child1 (512) + child2 (256) = 1792
        assert result == 1792

    def test_dedupes_shared_pids(self):
        """If a child appears in multiple process trees, count its RSS only once."""
        from sampler.metrics import sum_verify_rss

        shared_child = FakeProc(999, 'bash', [], rss=100)
        parent1 = FakeProc(1, 'bash', ['verify.sh'], rss=200, children=[shared_child])
        parent2 = FakeProc(2, 'bash', ['verify.sh'], rss=300, children=[shared_child])

        result = sum_verify_rss([parent1, parent2])
        # parent1 (200) + parent2 (300) + shared_child counted once (100) = 600
        assert result == 600

    def test_no_verify_processes_returns_zero(self):
        from sampler.metrics import sum_verify_rss

        procs = [FakeProc(1, 'python', ['python'], rss=1000)]
        result = sum_verify_rss(procs)
        assert result == 0

    def test_verify_without_children(self):
        from sampler.metrics import sum_verify_rss

        proc = FakeProc(1, 'bash', ['verify.sh'], rss=4096)
        result = sum_verify_rss([proc])
        assert result == 4096


# ---------------------------------------------------------------------------
# Step-1 (plan) tests: collect_process_metrics propagation on total failure
# ---------------------------------------------------------------------------


class TestCollectProcessMetricsDegrade:
    def test_total_proc_iter_failure_propagates(self):
        """A total proc_iter failure must propagate, not be swallowed into 0.0.

        Current code wraps proc_iter in `except Exception: procs = []` which
        converts a RuntimeError scan failure into a fabricated healthy-zero dict.
        This test confirms the RED premise: the exception is NOT raised.
        """
        from sampler.metrics import collect_process_metrics

        def raising(*_a, **_kw):
            raise RuntimeError('psutil scan down')

        with pytest.raises(RuntimeError):
            collect_process_metrics(proc_iter=raising, fd9_exists=lambda _pid: False)


# ---------------------------------------------------------------------------
# Task 2326 (DA1) step-9: anti-drift guard — parser re-homed to shared.psi
# ---------------------------------------------------------------------------


class TestParserRehomedToShared:
    def test_parse_pressure_file_is_shared_psi_object(self):
        """sampler.metrics.parse_pressure_file must be the SAME object as
        shared.psi.parse_pressure_file -- a re-export, not a duplicate copy
        that could drift and re-derive the kernel-asymmetry bug (DA-D9).

        This covers every caller in the module, /proc/pressure/* and each
        discovered cgroup's cpu.pressure alike: the cgroup text goes through
        α's parser, not a copy of it (INV-5).
        """
        import shared.psi

        import sampler.metrics

        assert sampler.metrics.parse_pressure_file is shared.psi.parse_pressure_file

    def test_collect_psi_default_reader_is_shared_read_pressure(self):
        import inspect

        import shared.psi

        from sampler.metrics import collect_psi

        sig = inspect.signature(collect_psi)
        assert sig.parameters['read'].default is shared.psi.read_pressure

    def test_collect_psi_still_returns_six_key_dict(self):
        """Behavior-identical: collect_psi's public contract is unaffected by the re-home."""
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}

        def read(name: str) -> str:
            return mapping[name]

        result = collect_psi(read=read)
        expected_keys = {
            'psi_cpu_some_avg10',
            'psi_cpu_full_avg10',
            'psi_mem_some_avg10',
            'psi_mem_full_avg10',
            'psi_io_some_avg10',
            'psi_io_full_avg10',
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Task 3592 step-1: discover_pressure_cgroups — the anchored per-leaf search
# ---------------------------------------------------------------------------


class TestDiscoverPressureCgroups:
    """PRD §6.4: one row per df-*.slice present, ELSE per orchestrator-*.service."""

    def test_df_topology_returns_every_leaf_sorted(self, tmp_path):
        from sampler.metrics import discover_pressure_cgroups

        tree = df_topology(tmp_path, 'reify', 'dark_factory')

        found = discover_pressure_cgroups(
            own_cgroup_path=tree.own_cgroup_path, cgroup_root=tree.cgroup_root
        )

        # Named by the LEAF DIRECTORY name, and sorted so the emitted metric
        # set is stable tick to tick rather than filesystem-order dependent.
        assert [name for name, _ in found] == [
            'df-dark_factory.slice',
            'df-reify.slice',
        ]
        for name, pressure_path in found:
            assert pressure_path.name == 'cpu.pressure'
            assert pressure_path.parent.name == name
            assert pressure_path.read_text().startswith('some avg10=')

    def test_live_topology_returns_all_seven_orchestrator_leaves(self, tmp_path):
        """The ELSE branch of PRD §6.4 — and the branch that is LIVE today.

        Measured on this host 2026-09-13: zero df-*.slice (task 3394 has not
        landed) and seven orchestrator-*.service leaves.
        """
        from sampler.metrics import discover_pressure_cgroups

        tree = live_topology(tmp_path)

        found = discover_pressure_cgroups(
            own_cgroup_path=tree.own_cgroup_path, cgroup_root=tree.cgroup_root
        )

        assert [name for name, _ in found] == sorted(LIVE_ORCHESTRATOR_LEAVES)
        assert len(found) == 7

    def test_a_near_miss_sibling_is_not_swept_in(self, tmp_path):
        """The SUFFIX half of both globs, which nothing else here exercises.

        Every other fixture plants only leaves that match, so relaxing the two
        globs to ``*/df-*`` and ``*/orchestrator-*`` left the suite green — and
        a cgroup manager plants more than services beside a service:
        ``orchestrator-*.socket`` and ``.scope`` units live in the same
        directory. Sweeping one in would emit ``own_cpu_some10:<name>`` rows
        named after a cgroup that is not the unit the arm is about, and ε2
        would average them in without a way to tell.

        Both branches get a decoy, because the two globs are separate spellings
        and only the branch that RUNS is under test in any one call.
        """
        from sampler.metrics import discover_pressure_cgroups

        orchestrator_tree = build_cgroup_tree(
            tmp_path / 'else-branch',
            groups={APP_PARENT: {
                'orchestrator-1.service': pressure_text(some=1.0),
                'orchestrator-1.socket': pressure_text(some=2.0),
                'orchestrator-1.scope': pressure_text(some=3.0),
            }},
        )
        found = discover_pressure_cgroups(
            own_cgroup_path=orchestrator_tree.own_cgroup_path,
            cgroup_root=orchestrator_tree.cgroup_root,
        )
        assert [name for name, _ in found] == ['orchestrator-1.service']

        df_tree = build_cgroup_tree(
            tmp_path / 'df-branch',
            groups={DF_PARENT: {
                'df-dark_factory.slice': pressure_text(some=1.5),
                'df-dark_factory': pressure_text(some=2.5),
            }},
        )
        found = discover_pressure_cgroups(
            own_cgroup_path=df_tree.own_cgroup_path,
            cgroup_root=df_tree.cgroup_root,
        )
        assert [name for name, _ in found] == ['df-dark_factory.slice']

    def test_df_wins_outright_when_both_topologies_exist(self, tmp_path):
        """df-*.slice is preferred, not unioned — the PRD says ELSE, not AND."""
        from sampler.metrics import discover_pressure_cgroups

        tree = build_cgroup_tree(
            tmp_path,
            groups={
                DF_PARENT: {'df-dark_factory.slice': pressure_text(some=1.5)},
                APP_PARENT: {
                    leaf: pressure_text(some=2.5) for leaf in LIVE_ORCHESTRATOR_LEAVES
                },
            },
        )

        found = discover_pressure_cgroups(
            own_cgroup_path=tree.own_cgroup_path, cgroup_root=tree.cgroup_root
        )

        assert [name for name, _ in found] == ['df-dark_factory.slice']

    def test_search_is_anchored_at_the_user_manager_segment(self, tmp_path):
        """A df-*.slice outside the own cgroup's user@NNN subtree is invisible.

        This is what makes the search cost 0.30 ms instead of the 32.7 ms a
        recursive rglob from the cgroup root measured on this host — and it is
        also a correctness property: another user's slices are not ours.
        """
        from sampler.metrics import discover_pressure_cgroups

        tree = build_cgroup_tree(
            tmp_path,
            groups={APP_PARENT: {leaf: pressure_text(some=0.5) for leaf in LIVE_ORCHESTRATOR_LEAVES}},
            outside={
                'df-someone_else.slice': pressure_text(some=9.0),
                'orchestrator-someone-else.service': pressure_text(some=9.0),
            },
        )

        found = discover_pressure_cgroups(
            own_cgroup_path=tree.own_cgroup_path, cgroup_root=tree.cgroup_root
        )

        names = [name for name, _ in found]
        assert 'df-someone_else.slice' not in names
        assert 'orchestrator-someone-else.service' not in names
        assert names == sorted(LIVE_ORCHESTRATOR_LEAVES)

    def test_outside_only_tree_finds_nothing_at_all(self, tmp_path):
        """With every candidate planted outside the anchor, the result is empty.

        Pins that discovery never falls back to walking the whole cgroup root:
        a search that did would return both planted leaves here.
        """
        from sampler.metrics import discover_pressure_cgroups

        tree = build_cgroup_tree(
            tmp_path,
            groups={},
            outside={'df-someone_else.slice': pressure_text(some=9.0)},
        )

        assert discover_pressure_cgroups(
            own_cgroup_path=tree.own_cgroup_path, cgroup_root=tree.cgroup_root
        ) == []

    @pytest.mark.parametrize(
        'own_cgroup_path',
        [
            pytest.param('/system.slice/some-daemon.service', id='no-user-manager-segment'),
            pytest.param('', id='alpha-resolution-failure-value'),
        ],
    )
    def test_unanchorable_own_cgroup_yields_empty_without_raising(
        self, tmp_path, own_cgroup_path
    ):
        """``''`` is α's OwnCgroup('', None) failure value — it must not raise."""
        from sampler.metrics import discover_pressure_cgroups

        tree = live_topology(tmp_path)

        assert discover_pressure_cgroups(
            own_cgroup_path=own_cgroup_path, cgroup_root=tree.cgroup_root
        ) == []

    def test_fixture_kernel_path_agrees_with_alphas_resolver(self, tmp_path):
        """The fixture's 0:: text and its directory layout are the same tree.

        Drives α's own seam (shared.psi.resolve_own_cgroup) over the fixture's
        proc_cgroup_path and asserts it produces the own_cgroup_path discovery
        is given, so the two halves of the fixture cannot drift apart.
        """
        from shared.psi import resolve_own_cgroup

        tree = live_topology(tmp_path)
        resolve_own_cgroup.cache_clear()
        own = resolve_own_cgroup(
            None, proc_cgroup_path=tree.proc_cgroup_path, cgroup_root=tree.cgroup_root
        )

        assert own.path == tree.own_cgroup_path
        assert own.pressure_path is not None
        assert own.pressure_path.read_text().startswith('some avg10=')


# ---------------------------------------------------------------------------
# Task 3592 step-3: collect_load_metrics — the runqueue half (PRD detail A)
# ---------------------------------------------------------------------------


class TestCollectLoadMetricsRunqueue:
    """After D1, runqueue_ratio is the only LIVE CPU arm.

    So an unreadable /proc/stat must leave a counter behind rather than
    silently making the arm inert: ``runqueue_read_ok`` is always emitted, and
    ``runqueue_ratio`` is emitted only when the read actually succeeded.
    """

    @staticmethod
    def _reader(reading):
        def read_runqueue(**_kwargs):
            return reading
        return read_runqueue

    def test_healthy_read_emits_ratio_and_ok(self, tmp_path):
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        tree = live_topology(tmp_path)
        result = collect_load_metrics(
            read_runqueue=self._reader(RunqueueReading(2.75, True)),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        )

        assert result['runqueue_ratio'] == pytest.approx(2.75)
        assert result['runqueue_read_ok'] == 1.0

    def test_failed_read_emits_ok_zero_and_no_ratio(self, tmp_path):
        """α degrades to RunqueueReading(0.0, False) — a value, not a reading.

        Persisting that 0.0 under ``runqueue_ratio`` would put a fabricated
        "completely idle host" row in the corpus ε1/ε2 calibrate against, which
        is the defect class task 1817 fixed in this module. The read_ok row
        alone carries the failure.
        """
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        tree = live_topology(tmp_path)
        result = collect_load_metrics(
            read_runqueue=self._reader(RunqueueReading(0.0, False)),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        )

        assert result['runqueue_read_ok'] == 0.0
        assert 'runqueue_ratio' not in result

    def test_a_genuine_zero_ratio_is_still_recorded(self, tmp_path):
        """A real 0.0 reading is distinguishable from a failed one: read_ok=1."""
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        tree = live_topology(tmp_path)
        result = collect_load_metrics(
            read_runqueue=self._reader(RunqueueReading(0.0, True)),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        )

        assert result['runqueue_ratio'] == 0.0
        assert result['runqueue_read_ok'] == 1.0

    @pytest.mark.parametrize('read_ok', [True, False])
    def test_every_value_is_a_float(self, tmp_path, read_ok):
        """The store's value column is REAL, so a bool read_ok must not leak."""
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        tree = live_topology(tmp_path)
        result = collect_load_metrics(
            read_runqueue=self._reader(RunqueueReading(1.25, read_ok)),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        )

        for key, value in result.items():
            assert type(value) is float, f'{key} is {type(value).__name__}: {value!r}'

    def test_default_reader_is_alphas_read_runqueue_ratio(self):
        """No /proc/stat reading of our own — α owns that reader (INV-5)."""
        import inspect

        import shared.psi

        from sampler.metrics import collect_load_metrics

        sig = inspect.signature(collect_load_metrics)
        assert sig.parameters['read_runqueue'].default is shared.psi.read_runqueue_ratio


# ---------------------------------------------------------------------------
# Task 3592 step-5: collect_load_metrics — the per-cgroup own-pressure half
# ---------------------------------------------------------------------------


def _healthy_runqueue(**_kwargs):
    from shared.psi import RunqueueReading

    return RunqueueReading(1.0, True)


def _collect(tree, **overrides):
    from sampler.metrics import collect_load_metrics

    return collect_load_metrics(
        read_runqueue=_healthy_runqueue,
        own_cgroup_path=overrides.pop('own_cgroup_path', tree.own_cgroup_path),
        cgroup_root=overrides.pop('cgroup_root', tree.cgroup_root),
        **overrides,
    )


class TestCollectLoadMetricsOwnPressure:
    """One own_cpu_some10 + one own_read_ok row per DISCOVERED leaf, and no more.

    ``own_read_ok:<leaf>`` is per-LEAF evidence — it answers "did THIS
    cgroup's pressure file read". Every degradation is therefore carried by
    value against a real leaf name (INV-11), and never fabricated against an
    invented one.
    """

    def test_seven_leaves_yield_fourteen_keys_with_parsed_values(self, tmp_path):
        tree = live_topology(tmp_path)

        result = _collect(tree)

        own = {k: v for k, v in result.items() if k.startswith('own_')}
        assert len(own) == 14
        # live_topology gives leaf i a `some avg10` of 0.5 + i, in
        # LIVE_ORCHESTRATOR_LEAVES order.
        for i, leaf in enumerate(LIVE_ORCHESTRATOR_LEAVES):
            assert own[f'own_cpu_some10:{leaf}'] == pytest.approx(0.5 + i)
            assert own[f'own_read_ok:{leaf}'] == 1.0
        # The leaf tail is the FULL leaf directory name, verbatim.
        assert 'own_cpu_some10:orchestrator-dark-factory.service' in own

    def test_the_live_none_branch_resolves_its_own_cgroup_hermetically(self, tmp_path):
        """The branch production actually takes, driven without touching /proc.

        __main__ calls ``collect_load_metrics()`` with no arguments, so the
        ``own_cgroup_path is None`` path — ask α's resolver for the live 0::
        path — is the one that runs on every real tick. Every other test in
        this class injects ``own_cgroup_path`` explicitly and so skips it,
        which left the live wiring (does it pass the right argument? does it
        take ``.path``?) protected by nothing. Forwarding α's own
        ``proc_cgroup_path`` seam is what makes it drivable against a fixture
        instead of only against the real /proc/self/cgroup.
        """
        from shared.psi import RunqueueReading, resolve_own_cgroup

        from sampler.metrics import collect_load_metrics

        tree = live_topology(tmp_path)
        resolve_own_cgroup.cache_clear()

        result = collect_load_metrics(
            read_runqueue=lambda **_kwargs: RunqueueReading(1.0, True),
            proc_cgroup_path=tree.proc_cgroup_path,
            cgroup_root=tree.cgroup_root,
        )

        own = {k: v for k, v in result.items() if k.startswith('own_')}
        assert len(own) == 14, (
            'the live None branch discovered no leaves, so the resolved anchor '
            f'disagrees with the injected one; got {sorted(own)}'
        )
        for i, leaf in enumerate(LIVE_ORCHESTRATOR_LEAVES):
            assert own[f'own_cpu_some10:{leaf}'] == pytest.approx(0.5 + i)
            assert own[f'own_read_ok:{leaf}'] == 1.0

    def test_the_resolution_reads_the_injected_proc_file_not_the_live_one(
        self, tmp_path, caplog
    ):
        """What makes the test above non-vacuous, and it needs saying why.

        LIVE_OWN_CGROUP_PATH is byte-identical to this host's real
        /proc/self/cgroup 0:: line, and the anchor is derived from the
        ``user@<uid>.service`` segment alone — so on this host a run that
        IGNORED the seam and read the real /proc would derive the very same
        anchor and discover the fixture's leaves anyway. The positive test
        therefore cannot, by itself, show the seam is plumbed.

        This one can: the injected 0:: line carries no ``user@`` segment, so
        α's resolver has nothing to anchor at and discovery must report zero
        leaves. A run that read the live /proc instead would anchor
        successfully and emit fourteen own_* rows, failing here.
        """
        from shared.psi import RunqueueReading, resolve_own_cgroup

        from sampler.metrics import collect_load_metrics

        tree = live_topology(tmp_path)
        tree.proc_cgroup_path.write_text('0::/system.slice/not-a-user-manager.service\n')
        resolve_own_cgroup.cache_clear()

        # DEBUG, not WARNING: an unanchorable 0:: line is the structural case
        # and is logged below warning level so a 5 s oneshot stays quiet.
        with caplog.at_level(logging.DEBUG, logger='sampler.metrics'):
            result = collect_load_metrics(
                read_runqueue=lambda **_kwargs: RunqueueReading(1.0, True),
                proc_cgroup_path=tree.proc_cgroup_path,
                cgroup_root=tree.cgroup_root,
            )

        assert not [k for k in result if k.startswith('own_')], (
            'own_* rows were emitted from an unanchorable 0:: line, so the '
            'resolution read the live /proc/self/cgroup and not the seam'
        )
        assert 'no pressure cgroups discovered' in caplog.text
        # The runqueue group is independent and still reports.
        assert result['runqueue_read_ok'] == 1.0

    def test_absent_pressure_file_degrades_only_its_own_leaf(self, tmp_path):
        broken = 'orchestrator-reify.service'
        tree = live_topology(tmp_path, **{broken.replace('.', '_').replace('-', '_'): None})

        result = _collect(tree)

        assert result[f'own_read_ok:{broken}'] == 0.0
        assert f'own_cpu_some10:{broken}' not in result
        # The other six are untouched — the failure is isolated, not pooled.
        for leaf in LIVE_ORCHESTRATOR_LEAVES:
            if leaf == broken:
                continue
            assert result[f'own_read_ok:{leaf}'] == 1.0
            assert f'own_cpu_some10:{leaf}' in result

    def test_unparseable_pressure_file_behaves_exactly_like_an_absent_one(self, tmp_path):
        broken = 'orchestrator-reify.service'
        key = broken.replace('.', '_').replace('-', '_')
        tree = live_topology(tmp_path, **{key: GARBAGE_PRESSURE_TEXT})

        result = _collect(tree)

        assert result[f'own_read_ok:{broken}'] == 0.0
        assert f'own_cpu_some10:{broken}' not in result
        assert len([k for k in result if k.startswith('own_')]) == 13

    def test_zero_discovered_leaves_emit_no_own_rows_and_warn(self, tmp_path, caplog):
        """Decision 2: no leaf means no subject, so no row — and one WARNING.

        A synthesised ``own_read_ok:<invented>`` = 0.0 would put a key into
        the DB that ε2 would then average, so the assertion is on the ABSENCE
        of any own_ key rather than on a particular invented name.
        """
        tree = build_cgroup_tree(tmp_path, groups={})

        with caplog.at_level(logging.WARNING):
            result = _collect(tree)

        assert [k for k in result if k.startswith('own_')] == []
        # The runqueue half is unaffected — the two halves degrade separately.
        assert result['runqueue_read_ok'] == 1.0
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('user@1000.service' in msg for msg in warnings), (
            f'Expected a WARNING naming the attempted anchor; got: {warnings}'
        )

    def test_an_unanchorable_path_says_so_below_warning_level(self, tmp_path, caplog):
        """The same "no leaves" outcome, at two severities, chosen by CAUSE.

        Its sibling above resolves an anchor and finds it empty — a surprise,
        so WARNING. Here there is no ``user@<uid>.service`` segment to anchor
        at, which is the ordinary state of a dev box, a container, or any
        sampler outside a systemd user manager. At the unit's 5 s cadence
        warning on that would be 17,280 journal lines a day carrying no
        information after the first, burying the per-tick ``logger.exception``
        lines the three degrade handlers exist to surface.

        Asserted as "no WARNING, and the DEBUG record still names the path",
        not as "logger.debug was called": the operator-visible property is the
        severity the journal filters on, and the diagnosis must survive the
        downgrade rather than be silenced by it.
        """
        tree = build_cgroup_tree(tmp_path, groups={})

        with caplog.at_level(logging.DEBUG, logger='sampler.metrics'):
            result = _collect(tree, own_cgroup_path='/system.slice/nowhere.service')

        assert [k for k in result if k.startswith('own_')] == []
        assert result['runqueue_read_ok'] == 1.0
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any('nowhere.service' in msg for msg in debug), (
            f'the diagnosis was silenced rather than downgraded; got: {debug}'
        )

    def test_some_only_pressure_file_still_reads_ok(self, tmp_path):
        """A cgroup cpu.pressure may legitimately carry no ``full`` line.

        α's parser returns a dict in that case, not None, so this is a
        successful read and must not be reported as a failure.
        """
        leaf = 'orchestrator-reify.service'
        key = leaf.replace('.', '_').replace('-', '_')
        tree = live_topology(tmp_path, **{key: pressure_text(some=3.25, full=None)})

        result = _collect(tree)

        assert result[f'own_read_ok:{leaf}'] == 1.0
        assert result[f'own_cpu_some10:{leaf}'] == pytest.approx(3.25)

    def test_df_topology_names_rows_by_the_slice_leaf(self, tmp_path):
        tree = df_topology(tmp_path, 'dark_factory', 'reify')

        result = _collect(tree)

        assert result['own_read_ok:df-dark_factory.slice'] == 1.0
        assert result['own_read_ok:df-reify.slice'] == 1.0
        assert len([k for k in result if k.startswith('own_')]) == 4

    def test_every_value_is_a_float(self, tmp_path):
        tree = live_topology(tmp_path, orchestrator_reify_service=None)

        result = _collect(tree)

        for key, value in result.items():
            assert type(value) is float, f'{key} is {type(value).__name__}: {value!r}'


# ---------------------------------------------------------------------------
# Task 3592 step-7: PRD §6.4 / boundary row 11 — sampler-to-gate vocabulary parity
# ---------------------------------------------------------------------------


class TestArmMetricStemParity:
    """Every gate ARM has a recorded sampler stem, and every stem is emitted.

    The gate decides on arms; the sampler records metrics; ε1/ε2 calibrate one
    against the other. If those two vocabularies drift the calibration
    silently measures the wrong thing, so the correspondence gets an explicit
    home (sampler.metrics.ARM_METRIC_STEMS) and this test.

    The arm side is read from the LIVE artifact, shared.psi._ARMS, not from a
    transcribed list (INV-10) — a private name deliberately, because it IS the
    saturation truth table and a copy of it here would be the drift this test
    exists to catch.
    """

    @staticmethod
    def _arm_fields():
        from shared.psi import _ARMS

        return {arm.field for arm in _ARMS}

    def test_every_arm_has_a_stem(self):
        from sampler.metrics import ARM_METRIC_STEMS

        missing = self._arm_fields() - set(ARM_METRIC_STEMS)
        assert not missing, (
            f'shared.psi._ARMS has arms the sampler records no metric for: {missing}. '
            'Add them to sampler.metrics.ARM_METRIC_STEMS and emit them, or the '
            'gate will hold on a signal the calibration corpus cannot see.'
        )

    def test_no_stem_is_orphaned(self):
        from sampler.metrics import ARM_METRIC_STEMS

        orphans = set(ARM_METRIC_STEMS) - self._arm_fields()
        assert not orphans, (
            f'ARM_METRIC_STEMS names keys that are not arms of shared.psi._ARMS: '
            f'{orphans}. The mapping must not grow a dead key.'
        )

    def test_psi_stems_are_actually_emitted_by_collect_psi(self):
        from sampler.metrics import ARM_METRIC_STEMS, collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        emitted = set(collect_psi(read=lambda name: mapping[name]))

        psi_stems = {
            stem for stem in ARM_METRIC_STEMS.values() if stem.startswith('psi_')
        }
        assert psi_stems, 'expected the four host-PSI arms to map to psi_* stems'
        assert psi_stems <= emitted, f'psi stems not emitted: {psi_stems - emitted}'

    def test_runqueue_stem_is_emitted_as_an_exact_key(self, tmp_path):
        from sampler.metrics import ARM_METRIC_STEMS

        tree = live_topology(tmp_path)
        emitted = set(_collect(tree))

        assert ARM_METRIC_STEMS['runqueue_ratio'] in emitted

    def test_own_cpu_stem_is_emitted_as_a_colon_prefix(self, tmp_path):
        from sampler.metrics import ARM_METRIC_STEMS

        tree = live_topology(tmp_path)
        emitted = set(_collect(tree))

        stem = ARM_METRIC_STEMS['own_cpu_some_avg10']
        assert any(key.startswith(f'{stem}:') for key in emitted), (
            f'no emitted key carries the stem {stem!r}; emitted: {sorted(emitted)}'
        )

# ---------------------------------------------------------------------------
# Task 3592 step-25: the arm-table LOCKSTEP guard (decision 7)
# ---------------------------------------------------------------------------


class TestCalibrationScriptArmTableLockstep:
    """One fact, two processes that cannot import each other, one reconciler.

    scripts/load-threshold-calibration.py runs under the system python3 at
    gate time and cannot import sampler.metrics, so it necessarily carries its
    own copy of the arm-to-metric correspondence. This test is what keeps the
    two copies equal.

    It lives in the SAMPLER suite because that is where both `sampler` and
    `shared` import reliably; scripts/tests runs under `--project shared`,
    where a probe showed sibling workspace members can be absent from the venv.
    """

    @staticmethod
    def _load_calibration_script():
        import importlib.util

        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / 'scripts' / 'load-threshold-calibration.py'
        spec = importlib.util.spec_from_file_location(
            'load_threshold_calibration', script
        )
        assert spec is not None, f'Could not build spec from {script}'
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def test_the_script_exposes_a_public_arm_to_selector_mapping(self):
        module = self._load_calibration_script()

        assert hasattr(module, 'ARM_METRIC_SELECTORS'), (
            'scripts/load-threshold-calibration.py must expose its arm table as one '
            'public module-level mapping, not as selectors inlined at their use sites '
            '— otherwise there is nothing for the lockstep test below to compare.'
        )
        for arm, spec in module.ARM_METRIC_SELECTORS.items():
            assert hasattr(spec, 'selector'), (
                f'ARM_METRIC_SELECTORS[{arm!r}] must be a spec carrying a named '
                '`selector`, so the ladder and unit label can live beside it without '
                f'the lockstep projection depending on tuple order; got {spec!r}'
            )

    def test_the_two_arm_tables_are_equal_in_both_directions(self):
        from sampler.metrics import ARM_METRIC_STEMS

        module = self._load_calibration_script()
        # Projected through the NAMED field rather than a position, so the
        # script's spec can carry its ladder and unit label beside the
        # selector without this test having to know their order.
        script_stems = {
            arm: spec.selector for arm, spec in module.ARM_METRIC_SELECTORS.items()
        }

        assert script_stems == ARM_METRIC_STEMS, (
            'DRIFT ALARM, not a bug in either file alone.\n'
            'sampler/src/sampler/metrics.py::ARM_METRIC_STEMS and '
            'scripts/load-threshold-calibration.py::ARM_METRIC_SELECTORS disagree.\n'
            f'  sampler: {ARM_METRIC_STEMS}\n'
            f'  script:  {script_stems}\n'
            'The SAMPLER owns the metric vocabulary — it is what writes the rows — so '
            'edit the script to match it, unless the sampler is what changed. They '
            'cannot be merged into one home: the script runs under the system python3 '
            'at gate time and cannot import sampler.'
        )

    def test_the_script_reads_a_corpus_THIS_STORE_wrote(self, tmp_path):
        """The THIRD copy of the samples schema, and until now the unguarded one.

        The schema is spelled in sampler.store (the writer), in
        dashboard/.../load.py (guarded by its own reconciler) and a third time
        as a fixture in scripts/tests/test_load_threshold_calibration.py, which
        cannot import sampler at all. That fixture is what every calibration
        test reads from, so the whole suite can stay green against a DDL the
        real store stopped writing years ago, and the first sign would be the
        gate returning `no_samples_in_window` against the live corpus.

        Guarded from the WRITER's side rather than by comparing two DDL strings.
        A string comparison would pin the schema's SPELLING -- it would go red
        on a reformatted CREATE TABLE and stay green if the script selected a
        column the store never had. This drives the script's own reader over a
        DB the real store created and wrote, so it fails on exactly the drift
        that matters and on nothing else.
        """
        from sampler.store import LoadSampleStore

        module = self._load_calibration_script()
        db_path = tmp_path / 'corpus.db'
        store = LoadSampleStore(db_path)
        for i in range(3):
            store.write_tick(
                1_000_000 + i * 5,
                unwindowed={'psi_cpu_some_avg10': 10.0 + i},
                windowed={'runqueue_ratio': 1.0 + i, 'runqueue_read_ok': 1.0},
            )

        series, readability, degradations = module.read_series(db_path, None)

        assert degradations == [], degradations
        assert [v for _, v in series['runqueue_ratio']] == [1.0, 2.0, 3.0]
        assert [v for _, v in series['psi_cpu_some_avg10']] == [10.0, 11.0, 12.0]
        assert [v for _, v in readability['runqueue_read_ok']] == [1.0, 1.0, 1.0]

    def test_every_declared_readability_metric_is_one_the_collectors_emit(
        self, tmp_path
    ):
        """The OTHER half of the arm table, which the projection above misses.

        ``test_the_two_arm_tables_are_equal_in_both_directions`` projects only
        ``spec.selector``, so ``spec.readability`` — the ``*_read_ok`` metric
        each arm's coverage row is computed from — has no reconciler at all.
        Renaming ``own_read_ok`` in sampler.metrics leaves every suite green
        and turns every ``own_*`` coverage row in the ε1/ε2 report into a
        phantom verdict about a metric nothing writes.

        Reconciled against what the REAL collectors EMIT rather than against
        another table, because the read_ok names are emitted as literals by
        ``collect_load_metrics`` and no table owns them.
        """
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        module = self._load_calibration_script()
        tree = live_topology(tmp_path)
        emitted = set(collect_load_metrics(
            read_runqueue=lambda **_kwargs: RunqueueReading(2.5, True),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        ))
        emitted_stems = {name.split(':', 1)[0] for name in emitted}

        declared = {
            spec.readability
            for spec in module.ARM_METRIC_SELECTORS.values()
            if spec.readability is not None
        }
        assert declared, (
            'no arm declares a readability metric; the coverage rows in the '
            'calibration report would all be "unknown"'
        )
        assert declared <= emitted_stems, (
            'DRIFT ALARM. scripts/load-threshold-calibration.py declares '
            f'readability metrics {sorted(declared - emitted_stems)} that '
            'sampler.metrics.collect_load_metrics does not emit.\n'
            f'  emitted: {sorted(emitted_stems)}\n'
            'Every coverage row computed from a missing metric reports on zero '
            'ticks, which is a fact about the report, not about the corpus.'
        )

    def test_every_emitted_read_ok_metric_is_claimed_by_an_arm(self, tmp_path):
        """The reverse direction: a NEW readability metric must not go unread.

        Without this, adding a ``*_read_ok`` collector and forgetting to point
        an ArmSpec at it silently produces an arm reported as having no
        readability metric at all — indistinguishable from the four host-PSI
        arms, which genuinely have none.
        """
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        module = self._load_calibration_script()
        tree = live_topology(tmp_path)
        emitted = set(collect_load_metrics(
            read_runqueue=lambda **_kwargs: RunqueueReading(2.5, True),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        ))
        emitted_read_ok = {
            name.split(':', 1)[0] for name in emitted if '_read_ok' in name
        }

        declared = {
            spec.readability
            for spec in module.ARM_METRIC_SELECTORS.values()
            if spec.readability is not None
        }
        assert emitted_read_ok <= declared, (
            'DRIFT ALARM. sampler.metrics emits readability metrics '
            f'{sorted(emitted_read_ok - declared)} that no ArmSpec claims, so '
            'nothing in the calibration report is computed from them.'
        )

    @pytest.mark.parametrize('read_ok', [True, False])
    def test_the_scripts_tick_clock_is_emitted_on_every_completed_tick(
        self, tmp_path, read_ok
    ):
        """Every coverage row's DENOMINATOR is this metric's row count.

        So it must be written on every tick ``collect_load_metrics`` completes
        — with no leaf discovered and with /proc/stat unreadable alike. Were it
        per-leaf or per-successful-read, the script would divide by the ticks
        something happened to be there, and report full coverage for a leaf
        observed over a fraction of the corpus.
        """
        from shared.psi import RunqueueReading

        from sampler.metrics import collect_load_metrics

        module = self._load_calibration_script()
        tree = build_cgroup_tree(tmp_path, groups={})
        emitted = collect_load_metrics(
            read_runqueue=lambda **_kwargs: RunqueueReading(0.0, read_ok),
            own_cgroup_path=tree.own_cgroup_path,
            cgroup_root=tree.cgroup_root,
        )

        assert module.TICK_METRIC in emitted, (
            f'scripts/load-threshold-calibration.py::TICK_METRIC '
            f'({module.TICK_METRIC!r}) is missing from a completed tick with no '
            f'leaves and read_ok={read_ok}; emitted: {sorted(emitted)}'
        )

    def test_loading_the_script_needs_no_first_party_package(self):
        """The property that makes this lockstep test possible at all."""
        module = self._load_calibration_script()

        for name in ('shared', 'sampler', 'orchestrator', 'yaml'):
            assert not hasattr(module, name), (
                f'{name} is bound at module level in the calibration script; it must '
                'stay stdlib-only so it loads under the system python3 at gate time.'
            )


# ---------------------------------------------------------------------------
# Review suggestion 3: the metrics <-> sampler vocabulary reconciler
# ---------------------------------------------------------------------------


class TestEmittedNamesAreAdmittedByRunTick:
    """The one pair of vocabularies inside this package with no reconciler.

    sampler.metrics EMITS metric names; sampler.sampler ADMITS them, against
    its own frozensets, with an ``assert`` placed BEFORE every insert. So a
    mismatch does not degrade one group — it aborts the whole tick, and the
    sampler writes zero rows of any group every 5 s until someone reads the
    journal.

    Nothing caught that. The lockstep guards added by this task reconcile
    metrics.py against the calibration script, and _ARMS against
    ARM_METRIC_STEMS, but every run_tick test feeds a hand-written dict and
    every __main__ test monkeypatches the collectors away. Renaming an emitted
    stem consistently across metrics.py, ARM_METRIC_STEMS and the calibration
    script therefore left the whole suite green and production broken.

    This drives the REAL collectors into the REAL run_tick, through public
    interfaces only — no private frozenset is imported here, because a row
    landing in the store is the property that actually matters.
    """

    @staticmethod
    def _collect_all(tree):
        from shared.psi import RunqueueReading

        from sampler.metrics import (
            collect_load_metrics,
            collect_process_metrics,
            collect_psi,
        )

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        return {
            'psi': collect_psi(read=lambda name: mapping[name]),
            'process_metrics': collect_process_metrics(
                proc_iter=lambda _attrs: iter(()), fd9_exists=lambda _pid: False
            ),
            'load_metrics': collect_load_metrics(
                read_runqueue=lambda **_kwargs: RunqueueReading(2.5, True),
                own_cgroup_path=tree.own_cgroup_path,
                cgroup_root=tree.cgroup_root,
            ),
        }

    def test_every_name_the_collectors_emit_reaches_the_store(self, tmp_path):
        import sqlite3

        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        tree = live_topology(tmp_path)
        groups = self._collect_all(tree)
        emitted = {name for group in groups.values() for name in group}
        # 14 own_* (7 leaves x value+read_ok) + 2 runqueue + 6 psi + 3 process
        # — the 25 metrics/tick the retention sizing was measured against.
        assert len(emitted) == 14 + 2 + 6 + 3, (
            f'the fixture did not exercise all three groups; got {sorted(emitted)}'
        )

        db_path = tmp_path / 'db.sqlite'
        # Raises AssertionError out of run_tick if any emitted name is one the
        # guard does not admit — which is exactly the production failure.
        run_tick(LoadSampleStore(db_path), 1_000_000, **groups)

        conn = sqlite3.connect(str(db_path))
        try:
            stored = {row[0] for row in conn.execute('SELECT DISTINCT metric FROM samples')}
        finally:
            conn.close()
        assert stored == emitted

    def test_the_df_topology_vocabulary_is_admitted_too(self, tmp_path):
        """Post-3394 the leaf names change shape; the stem must still admit them."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        tree = df_topology(tmp_path, 'dark-factory', 'reify')
        groups = self._collect_all(tree)
        load = groups['load_metrics']
        assert 'own_cpu_some10:df-dark-factory.slice' in load, sorted(load)

        run_tick(LoadSampleStore(tmp_path / 'db.sqlite'), 1_000_000, **groups)
