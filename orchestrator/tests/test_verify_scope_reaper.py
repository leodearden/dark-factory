"""Tests for the leftover ``df-verify-*.scope`` startup reaper (task 2829).

Covers the project-tagged verify-scope naming and the tag-scoped startup
sweep that stops only THIS project's leftover verify scopes:

  step-1/2  — ``verify._scope_tag_for``: a deterministic, systemd-name-safe
              per-project tag (basename slug + path-hash suffix).
  step-3/4  — ``verify._verify_scope_name``: the tagged, uuid-suffixed
              ``df-verify-{tag}-<hex>.scope`` unit name.
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
import re
from pathlib import Path

from orchestrator import verify


class TestScopeTagFor:
    """``verify._scope_tag_for`` derives a deterministic, systemd-safe per-project tag."""

    def test_tag_is_systemd_name_safe_slug(self, tmp_path: Path) -> None:
        """The tag contains only lowercase ``[a-z0-9-]`` (embeddable in a scope
        unit name without escaping) and is non-empty."""
        tag = verify._scope_tag_for(tmp_path)
        assert tag, 'tag must be non-empty'
        assert re.fullmatch(r'[a-z0-9-]+', tag), (
            f'tag must contain only lowercase [a-z0-9-]; got {tag!r}'
        )

    def test_tag_is_stable_across_calls(self, tmp_path: Path) -> None:
        """Same project_root -> same tag, so a fresh boot reaps its dead
        predecessor's same-tagged scopes."""
        assert verify._scope_tag_for(tmp_path) == verify._scope_tag_for(tmp_path)

    def test_distinct_roots_yield_distinct_tags(self, tmp_path: Path) -> None:
        """Different project_roots get different tags."""
        root_a = tmp_path / 'alpha'
        root_b = tmp_path / 'beta'
        root_a.mkdir()
        root_b.mkdir()
        assert verify._scope_tag_for(root_a) != verify._scope_tag_for(root_b)

    def test_same_basename_different_path_disambiguated_by_hash(
        self, tmp_path: Path,
    ) -> None:
        """Two projects that SHARE a basename but live at different absolute
        paths still get distinct tags — the path-hash suffix disambiguates, so
        one project's sweep can never match the other's scopes."""
        a = tmp_path / 'a' / 'proj'
        b = tmp_path / 'b' / 'proj'
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        tag_a = verify._scope_tag_for(a)
        tag_b = verify._scope_tag_for(b)
        assert tag_a != tag_b, (
            'same-basename roots must differ via the path-hash suffix; '
            f'got {tag_a!r} == {tag_b!r}'
        )
        # Both still carry the shared basename slug for operator legibility.
        assert 'proj' in tag_a and 'proj' in tag_b


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
