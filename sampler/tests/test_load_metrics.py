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
    def test_both_lines_extracted(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_CPU_TEXT)
        assert result == {'some_avg10': 2.50, 'full_avg10': 0.30}

    def test_missing_full_defaults_to_zero(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_MEM_TEXT)
        assert result == {'some_avg10': 1.23, 'full_avg10': 0.0}

    def test_io_both_lines(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_IO_TEXT)
        assert result == {'some_avg10': 0.75, 'full_avg10': 0.45}

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
