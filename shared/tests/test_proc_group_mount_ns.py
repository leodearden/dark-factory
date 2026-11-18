"""The at-or-under scans translate paths through mount namespaces (PRD C11, decision 16).

``plans/canonical-build-path-shared-cache-prd.md`` decision 16: a process in another mount
namespace that sees a lane at a bind-mount point names it by that mount point, so a scan
must translate the path back before matching. The real-namespace classes build boundary
row 19 with :mod:`shared.testing_mount_ns`; the synthetic classes pin every branch against
a fabricated /proc through ``shared.proc_group._PROC_ROOT``, the idiom of
``test_proc_group.py::TestScanProcessGroupsAgainstASyntheticProc``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from _synthetic_proc import synthetic_stat_line

import shared.proc_group as proc_group_module
from shared.proc_group import pid_running, reap_process_groups, scan_process_groups_under_path
from shared.testing_mount_ns import Hold, namespaced_holder, unreadable_foreign_namespace_process

_MOUNT_POINT = '/opt/canon/ws'


class _FabricatedNamespaces:
    """A fabricated /proc whose pids live in fabricated mount namespaces.

    A namespace's identity is the inode of a regular file under ``ns/``, and
    ``proc/self/ns/mnt`` points at ``host``. A fake namespace root holds the mount point as a
    symlink to the lane, which ``os.stat`` follows to the lane's device and inode: it stands
    in for the bind. ``elsewhere_root`` holds the same mount point aimed at another directory.
    """

    def __init__(self, base: Path) -> None:
        self.proc_root = base / 'proc'
        self.lane = base / 'lane'
        (self.lane / 'sub').mkdir(parents=True)
        elsewhere = base / 'elsewhere'
        elsewhere.mkdir()
        namespaces = base / 'ns'
        namespaces.mkdir()
        self.host, self.sandbox, self.other = (namespaces / n for n in ('host', 'sandbox', 'other'))
        for identity in (self.host, self.sandbox, self.other):
            identity.write_text('')
        self.sandbox_root = self._namespace_root(base / 'nsroot', self.lane)
        self.elsewhere_root = self._namespace_root(base / 'nsroot-elsewhere', elsewhere)
        lane_dev = os.stat(self.lane).st_dev
        self.lane_device = (os.major(lane_dev), os.minor(lane_dev))
        (self.proc_root / 'self' / 'ns').mkdir(parents=True)
        (self.proc_root / 'self' / 'ns' / 'mnt').symlink_to(self.host)

    @staticmethod
    def _namespace_root(root: Path, target: Path) -> Path:
        for mount_point in ('opt/canon/ws', 'opt/canon ws'):
            (root / mount_point).parent.mkdir(parents=True, exist_ok=True)
            (root / mount_point).symlink_to(target)
        return root

    def mountinfo(self, mount_point_field: str, *, device: tuple[int, int] | None = None) -> str:
        """The namespace's '/' plus one bind of the lane at *mount_point_field* (escaped form)."""
        major, minor = device or self.lane_device
        return (
            f'21 1 {major}:{minor} / / rw - ext4 /dev/x rw\n'
            f'36 21 {major}:{minor} /lane {mount_point_field} rw - ext4 /dev/x rw\n'
        )

    def add_pid(
        self,
        pid: int,
        *,
        ns: Path | None,
        root: Path | None,
        mountinfo: str,
        cwd: str | None = None,
        fds: Sequence[str] = (),
        maps: Sequence[str] = (),
        cpu: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        """Plant pid *pid* in its own group; a None *ns* or *root* leaves that entry missing."""
        entry = self.proc_root / str(pid)
        entry.mkdir()
        (entry / 'stat').write_text(synthetic_stat_line(pid, pid, cpu=cpu))
        (entry / 'mountinfo').write_text(mountinfo)
        if ns is not None:
            (entry / 'ns').mkdir()
            (entry / 'ns' / 'mnt').symlink_to(ns)
        if root is not None:
            (entry / 'root').symlink_to(root)
        if cwd is not None:
            (entry / 'cwd').symlink_to(cwd)
        (entry / 'fd').mkdir()
        for number, target in enumerate(fds, start=3):
            (entry / 'fd' / str(number)).symlink_to(target)
        (entry / 'maps').write_text(
            ''.join(f'7f0a1d000000-7f0a1d002000 r-xp 00000000 08:01 4242       {p}\n' for p in maps)
        )

    def add_sandboxed(self, pid: int, **signals) -> None:
        """A pid of the sandbox namespace, where the lane is bound at ``/opt/canon/ws``."""
        self.add_pid(
            pid,
            ns=self.sandbox,
            root=self.sandbox_root,
            mountinfo=self.mountinfo(_MOUNT_POINT),
            **signals,
        )


@pytest.fixture
def namespaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FabricatedNamespaces:
    fabricated = _FabricatedNamespaces(tmp_path.resolve())
    monkeypatch.setattr('shared.proc_group._PROC_ROOT', fabricated.proc_root)
    return fabricated


class TestScanThroughARealMountNamespace:
    """Boundary row 19 against the real /proc: one test, because each real walk costs seconds."""

    def test_holders_through_a_bind_are_found_and_the_unreadable_namespace_is_spared(
        self, tmp_path: Path
    ) -> None:
        base = tmp_path.resolve()
        lane = base / 'lane'
        mount_point = base / 'canon ws'
        elsewhere = base / 'elsewhere'
        elsewhere.mkdir()
        with contextlib.ExitStack() as stack:
            holders = {
                hold: stack.enter_context(namespaced_holder(lane, mount_point, hold))
                for hold in (Hold.CWD, Hold.OPEN_FILE, Hold.MAPPING)
            }
            unrelated = stack.enter_context(unreadable_foreign_namespace_process(cwd=elsewhere))

            found = scan_process_groups_under_path(lane)

            expected = {holder.pgid for holder in holders.values()}
            assert found == expected, (
                f'expected exactly the holders {holders} (unrelated {unrelated}); got {found}'
            )
            outcomes = reap_process_groups(found)
            assert outcomes == dict.fromkeys(expected, 'reaped')
            assert pid_running(unrelated.pid), 'the unreadable-namespace process was signalled'


class TestNamespaceTranslationAgainstASyntheticProc:
    """Every translation branch, against fabricated namespaces."""

    def test_cwd_fd_and_mapping_through_the_mount_point_are_found(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        namespaces.add_sandboxed(101, cwd=f'{_MOUNT_POINT}/sub')
        namespaces.add_sandboxed(102, cwd='/', fds=[f'{_MOUNT_POINT}/x'])
        namespaces.add_sandboxed(103, cwd='/', maps=[f'{_MOUNT_POINT}/lib.so'])

        assert scan_process_groups_under_path(namespaces.lane) == {101, 102, 103}

    def test_an_escaped_mount_point_is_unescaped(self, namespaces: _FabricatedNamespaces) -> None:
        namespaces.add_pid(
            101,
            ns=namespaces.sandbox,
            root=namespaces.sandbox_root,
            mountinfo=namespaces.mountinfo(r'/opt/canon\040ws'),
            cwd='/opt/canon ws/sub',
        )

        assert scan_process_groups_under_path(namespaces.lane) == {101}

    def test_the_lane_bound_as_the_namespace_root_covers_every_path(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        major, minor = namespaces.lane_device
        namespaces.add_pid(
            101,
            ns=namespaces.sandbox,
            root=namespaces.lane,
            mountinfo=f'36 21 {major}:{minor} /lane / rw - ext4 /dev/x rw\n',
            cwd='/sub',
        )

        assert scan_process_groups_under_path(namespaces.lane) == {101}

    def test_one_namespace_is_resolved_once_per_scan(
        self, namespaces: _FabricatedNamespaces, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        namespaces.add_sandboxed(101, cwd=f'{_MOUNT_POINT}/sub')
        namespaces.add_sandboxed(102, cwd=f'{_MOUNT_POINT}/sub')
        resolved: list[str] = []
        real = proc_group_module._mount_points_bound_to

        def counting(entry, identity):
            resolved.append(entry.name)
            return real(entry, identity)

        monkeypatch.setattr('shared.proc_group._mount_points_bound_to', counting)

        assert scan_process_groups_under_path(namespaces.lane) == {101, 102}
        assert len(resolved) == 1, f'the sandbox namespace was resolved for each of {resolved}'

    def test_a_sibling_of_the_mount_point_is_not_found(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        namespaces.add_sandboxed(101, cwd=f'{_MOUNT_POINT}/sub')
        namespaces.add_sandboxed(102, cwd=f'{_MOUNT_POINT}XYZ')

        assert scan_process_groups_under_path(namespaces.lane) == {101}

    def test_a_namespace_whose_root_cannot_be_read_gives_no_translation(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        """PRD decision 16: skipped, never fail-closed; pid 101 shows the scan did not raise."""
        namespaces.add_sandboxed(101, cwd=f'{_MOUNT_POINT}/sub')
        namespaces.add_pid(
            102,
            ns=namespaces.other,
            root=None,
            mountinfo=namespaces.mountinfo(_MOUNT_POINT),
            cwd=f'{_MOUNT_POINT}/sub',
        )

        assert scan_process_groups_under_path(namespaces.lane) == {101}

    def test_a_pid_in_the_scanners_own_namespace_is_not_translated(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        namespaces.add_sandboxed(101, cwd=f'{_MOUNT_POINT}/sub')
        namespaces.add_pid(
            102,
            ns=namespaces.host,
            root=namespaces.sandbox_root,
            mountinfo=namespaces.mountinfo(_MOUNT_POINT),
            cwd=f'{_MOUNT_POINT}/sub',
        )

        assert scan_process_groups_under_path(namespaces.lane) == {101}

    def test_a_pid_with_no_namespace_entry_is_matched_as_before(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        for pid, cwd in ((101, f'{_MOUNT_POINT}/sub'), (102, str(namespaces.lane / 'sub'))):
            namespaces.add_pid(
                pid,
                ns=None,
                root=namespaces.sandbox_root,
                mountinfo=namespaces.mountinfo(_MOUNT_POINT),
                cwd=cwd,
            )

        assert scan_process_groups_under_path(namespaces.lane) == {102}

    def test_a_namespaced_pid_naming_the_lanes_real_path_is_still_found(
        self, namespaces: _FabricatedNamespaces
    ) -> None:
        """PRD decision 7: the per-session binding shows the lane at both paths."""
        namespaces.add_sandboxed(101, cwd=str(namespaces.lane / 'sub'))

        assert scan_process_groups_under_path(namespaces.lane) == {101}

    @pytest.mark.parametrize(
        ('aimed_elsewhere', 'minor_offset'),
        [
            pytest.param(False, 1, id='another-maj-min-though-the-path-is-the-lane'),
            pytest.param(True, 0, id='the-lanes-maj-min-on-another-directory'),
        ],
    )
    def test_a_mount_that_is_not_the_lane_is_not_an_alias(
        self, namespaces: _FabricatedNamespaces, aimed_elsewhere: bool, minor_offset: int
    ) -> None:
        major, minor = namespaces.lane_device
        namespaces.add_pid(
            101,
            ns=namespaces.sandbox,
            root=namespaces.elsewhere_root if aimed_elsewhere else namespaces.sandbox_root,
            mountinfo=namespaces.mountinfo(_MOUNT_POINT, device=(major, minor + minor_offset)),
            cwd=f'{_MOUNT_POINT}/sub',
        )
        namespaces.add_pid(
            102, ns=None, root=None, mountinfo='', cwd=str(namespaces.lane / 'sub')
        )

        assert scan_process_groups_under_path(namespaces.lane) == {102}
