"""Tests for the leftover ``df-verify-*.scope`` startup reaper (task 2829).

Covers the project-tagged verify-scope naming and the tag-scoped startup
sweep that stops only THIS project's leftover verify scopes:

  step-3/4  — ``verify._verify_scope_name``: the tagged, uuid-suffixed
              ``df-verify-{tag}-<hex>.scope`` unit name, built through
              ``shared/src/shared/verify_scope_tag.py``, which also owns the
              per-project tag rule and its tests.
  step-5/6  — ``verify.reap_leftover_verify_scopes``: a TAG-SCOPED
              ``systemctl --user list-units`` enumeration + defensive
              re-filter + per-unit ``_kill_cgroup_scope`` reap, fully
              fail-soft.

The tag scoping is the cross-project-safety guarantee: all
``orchestrator-*.service`` units share ONE per-user ``systemctl --user``
session, so a bare-glob ``df-verify-*.scope`` sweep would reap a sibling
project's LIVE in-flight verify scope. Embedding a per-project tag confines
each orchestrator's sweep to its own leftovers.
"""
import logging
import re
from pathlib import Path

import pytest
from shared.verify_scope_tag import parse_verify_scope_unit, scope_tag_for

from orchestrator import verify


class TestVerifyScopeName:
    """``verify._verify_scope_name`` builds a tagged, uuid-suffixed scope unit name."""

    def test_name_has_prefix_tag_and_scope_suffix(self) -> None:
        """The name preserves the ``df-verify-`` prefix (so existing prefix
        matchers still work), embeds the tag segment, and ends in ``.scope``
        with a 12-hex uuid segment between the tag and the suffix."""
        tag = 'myproj-1a2b3c4d'
        name = verify._verify_scope_name(tag)
        assert name.startswith(f'df-verify-{tag}-'), name
        assert name.endswith('.scope'), name
        m = re.fullmatch(rf'df-verify-{re.escape(tag)}-([0-9a-f]{{12}})\.scope', name)
        assert m, f'unexpected scope-name shape: {name!r}'

    def test_each_call_has_unique_uuid_segment(self) -> None:
        """Every call yields a distinct uuid segment so concurrent verifies
        never collide on a scope unit name."""
        tag = 'myproj-1a2b3c4d'
        names = {verify._verify_scope_name(tag) for _ in range(20)}
        assert len(names) == 20, f'expected 20 unique scope names; got {len(names)}'

    def test_name_parses_with_the_shared_parser(self) -> None:
        """The name is built through the shared rule, so the load recorder's
        parser reads back exactly the tag and a 12-hex instance."""
        tag = 'myproj-1a2b3c4d'
        unit = parse_verify_scope_unit(verify._verify_scope_name(tag))
        assert unit is not None
        assert unit.tag == tag
        assert re.fullmatch(r'[0-9a-f]{12}', unit.instance), unit

    def test_a_tag_the_shared_parser_would_not_read_is_refused(self) -> None:
        with pytest.raises(ValueError, match='Bad Tag'):
            verify._verify_scope_name('Bad Tag')


class TestReapLeftoverVerifyScopes:
    """``verify.reap_leftover_verify_scopes`` enumerates + reaps ONLY this
    project's leftover verify scopes, fully fail-soft."""

    @staticmethod
    def _install_fake_exec(monkeypatch, listing: str) -> list[list[str]]:
        """Patch ``create_subprocess_exec`` to capture argv and feed *listing*
        to the ``list-units`` enumeration; also stub ``shutil.which`` truthy so
        the systemctl-availability gate passes. Returns the captured-argv list.
        """
        captured: list[list[str]] = []

        class _FakeProc:
            def __init__(self, stdout: bytes) -> None:
                self._stdout = stdout

            async def communicate(self):
                return (self._stdout, b'')

            async def wait(self):
                return 0

        async def fake_exec(*args, **kwargs):
            captured.append(list(args))
            if 'list-units' in args:
                return _FakeProc(listing.encode())
            return _FakeProc(b'')

        monkeypatch.setattr(
            'orchestrator.verify.asyncio.create_subprocess_exec', fake_exec,
        )
        monkeypatch.setattr(
            'orchestrator.verify.shutil.which', lambda name: f'/usr/bin/{name}',
        )
        return captured

    @pytest.mark.asyncio
    async def test_enumeration_is_tag_scoped_and_reaps_matching_units(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) enumeration glob is TAG-SCOPED (cross-project safety); (b) each
        matching unit gets kill+stop; (c) a sibling-tagged unit is defensively
        dropped; (d) returns the reaped names."""
        tag = scope_tag_for(tmp_path)
        unit_a = f'df-verify-{tag}-aaa111aaa111.scope'
        unit_b = f'df-verify-{tag}-bbb222bbb222.scope'
        # A DIFFERENTLY-tagged unit systemctl might over-return — must be dropped.
        sibling = 'df-verify-siblingproj-99887766-ccc333ccc333.scope'
        listing = (
            f'{unit_a} loaded active running Verify A\n'
            f'{unit_b} loaded active running Verify B\n'
            f'{sibling} loaded active running Sibling verify\n'
        )
        captured = self._install_fake_exec(monkeypatch, listing)

        reaped = await verify.reap_leftover_verify_scopes(tmp_path)

        # (a) TAG-SCOPED enumeration, NOT a bare df-verify-*.scope glob.
        enum_calls = [a for a in captured if 'list-units' in a]
        assert len(enum_calls) == 1, captured
        enum = enum_calls[0]
        assert enum[:2] == ['systemctl', '--user'], enum
        assert '--all' in enum and '--no-legend' in enum, enum
        assert f'df-verify-{tag}-*.scope' in enum, enum
        assert 'df-verify-*.scope' not in enum, (
            f'enumeration must be tag-scoped, not a bare glob; got {enum}'
        )
        # (b) each matching unit got kill + stop (via _kill_cgroup_scope).
        for unit in (unit_a, unit_b):
            assert [
                'systemctl', '--user', 'kill', '--signal=SIGKILL', unit,
            ] in captured, unit
            assert ['systemctl', '--user', 'stop', unit] in captured, unit
        # (c) the sibling-tagged unit is defensively dropped — NEVER touched.
        assert sibling not in reaped
        assert not any(sibling in argv for argv in captured), (
            f'sibling unit must never be passed to a kill/stop call; got {captured}'
        )
        # (d) returns the reaped names.
        assert reaped == [unit_a, unit_b], reaped

    @pytest.mark.asyncio
    async def test_sweep_is_keyed_by_the_shared_tag(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The enumeration glob is the shared rule's tag for the root, so the
        sweep and the load recorder key the same project the same way, even for
        a basename the slug has to sanitise."""
        root = tmp_path / 'My_Solar Challenge'
        root.mkdir()
        captured = self._install_fake_exec(monkeypatch, '')

        await verify.reap_leftover_verify_scopes(root)

        enum = next(a for a in captured if 'list-units' in a)
        assert enum[-1] == f'df-verify-{scope_tag_for(root)}-*.scope', enum

    @pytest.mark.asyncio
    async def test_surviving_scope_is_not_reported_and_is_warned(
        self, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        """A scope still ACTIVE after the reap attempt is CONFIRMED not gone:
        it is NOT returned as reaped and is surfaced loudly at WARNING, rather
        than being silently over-reported as reaped (a best-effort
        ``_kill_cgroup_scope`` can leave a genuinely un-killable scope alive)."""
        tag = scope_tag_for(tmp_path)
        dead = f'df-verify-{tag}-dead000dead0.scope'
        alive = f'df-verify-{tag}-alive0alive0.scope'
        listing = (
            f'{dead} loaded active running Verify dead\n'
            f'{alive} loaded active running Verify alive\n'
        )

        class _FakeProc:
            def __init__(self, stdout: bytes) -> None:
                self._stdout = stdout

            async def communicate(self):
                return (self._stdout, b'')

            async def wait(self):
                return 0

        async def fake_exec(*args, **kwargs):
            if 'list-units' in args:
                return _FakeProc(listing.encode())
            if 'is-active' in args:
                # `alive` stays active (the reap did not take); `dead` is gone.
                return _FakeProc(b'active\n' if args[-1] == alive else b'inactive\n')
            return _FakeProc(b'')  # kill / stop

        monkeypatch.setattr(
            'orchestrator.verify.asyncio.create_subprocess_exec', fake_exec,
        )
        monkeypatch.setattr(
            'orchestrator.verify.shutil.which', lambda name: f'/usr/bin/{name}',
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            reaped = await verify.reap_leftover_verify_scopes(tmp_path)

        # Only the CONFIRMED-gone scope is reported reaped.
        assert reaped == [dead], reaped
        assert alive not in reaped
        # The survivor is surfaced loudly; the reaped one is not warned about.
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(alive in m for m in warnings), (
            f'expected a WARNING naming the surviving scope {alive!r}; got {warnings}'
        )
        assert all(dead not in m for m in warnings), (
            f'a confirmed-reaped scope must not be warned about; got {warnings}'
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_systemctl_absent(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Fail-soft: no systemctl -> [] and no enumeration attempted."""
        called = False

        async def fake_exec(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError('must not enumerate when systemctl is absent')

        monkeypatch.setattr('orchestrator.verify.shutil.which', lambda name: None)
        monkeypatch.setattr(
            'orchestrator.verify.asyncio.create_subprocess_exec', fake_exec,
        )

        reaped = await verify.reap_leftover_verify_scopes(tmp_path)

        assert reaped == []
        assert not called

    @pytest.mark.asyncio
    async def test_fail_soft_when_subprocess_raises(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Fail-soft: a raising subprocess yields [] and never propagates."""

        async def fake_exec(*args, **kwargs):
            raise OSError('systemctl boom')

        monkeypatch.setattr(
            'orchestrator.verify.shutil.which', lambda name: f'/usr/bin/{name}',
        )
        monkeypatch.setattr(
            'orchestrator.verify.asyncio.create_subprocess_exec', fake_exec,
        )

        reaped = await verify.reap_leftover_verify_scopes(tmp_path)

        assert reaped == []
