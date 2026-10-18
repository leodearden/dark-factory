"""A corpus fixture's reviewed tree, materialised in a hermetic scratch clone.

Every test drives ``evals/reviewer_trial/checkout.py::fixture_checkout`` against
a real source repository under ``tmp_path``: what the reviewer sees is git's own
answer, so a fake would only test the fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.evals.reviewer_trial.checkout import fixture_checkout
from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    FixtureContext,
    FixtureTaskRecord,
)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


class _Source:
    """A source repo whose main moved on after the fixture's base commit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()
        _git(root, 'init', '-q', '-b', 'main')
        _git(root, 'config', 'user.email', 'source@example.com')
        _git(root, 'config', 'user.name', 'Source')
        _git(root, 'config', 'commit.gpgsign', 'false')
        self.base = _commit(root, {'a.py': 'x = 1\n', 'gui/g.py': 'g = 1\n'}, 'base')
        _commit(root, {'a.py': 'x = 2\n'}, 'main moves a.py on')
        self.head = _commit(root, {'b.py': 'b = 1\n'}, 'main adds b.py')


@pytest.fixture
def source(tmp_path: Path) -> _Source:
    return _Source(tmp_path / 'source')


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    return scratch


_CHANGE_AT_BASE = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x = 1
+x = 10
diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+n = 1
"""

_FICTIONAL_PRE_IMAGE = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x = 999
+x = 10
"""

_REAL_WORLD_FORMAT = """\
── a.py (modified) ──
-x = 1
+x = 10
"""

_MISCOUNTED_HUNK = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,4 +1,6 @@
-x = 1
+x = 10
+y = 20
"""

_GUI_CHANGE = """\
diff --git a/gui/g.py b/gui/g.py
--- a/gui/g.py
+++ b/gui/g.py
@@ -1 +1 @@
-g = 1
+g = 2
"""


def _fixture(
    diff_text: str,
    *,
    cwd: Path,
    base_sha: str | None,
    changed_files: list[str],
    base_sha_reason: str | None = None,
) -> CorpusDiff:
    return CorpusDiff(
        diff_id='fx_1',
        language='python',
        source='synthetic',
        diff_text=diff_text,
        description='a fixture',
        ground_truth=[],
        cwd=cwd,
        context=FixtureContext(
            task=FixtureTaskRecord(title='T', description='d', test_strategy='ts'),
            changed_files=changed_files,
            base_sha=base_sha,
            base_sha_reason=base_sha_reason,
        ),
    )


@pytest.mark.asyncio
class TestAppliedAtItsBase:
    async def test_head_is_the_change_committed_on_the_base(
        self, source: _Source, scratch: Path,
    ):
        diff = _fixture(
            _CHANGE_AT_BASE, cwd=source.root, base_sha=source.base,
            changed_files=['a.py', 'new.py'],
        )

        async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
            assert checkout.change_base == source.base
            assert checkout.unapplied_reason is None
            assert _git(checkout.root, 'rev-parse', 'HEAD^') == source.base
            changed = _git(
                checkout.root, 'diff', '--name-only', f'{source.base}...HEAD',
            ).splitlines()
            assert sorted(changed) == ['a.py', 'new.py']
            assert (checkout.root / 'a.py').read_text() == 'x = 10\n'
            assert (checkout.root / 'new.py').read_text() == 'n = 1\n'
            assert not (checkout.root / 'b.py').exists()

    async def test_the_clone_is_a_shared_scratch_copy_removed_on_exit(
        self, source: _Source, scratch: Path,
    ):
        diff = _fixture(
            _CHANGE_AT_BASE, cwd=source.root, base_sha=source.base,
            changed_files=['a.py', 'new.py'],
        )

        async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
            root = checkout.root
            assert root.resolve() != source.root.resolve()
            assert (root / '.git' / 'objects' / 'info' / 'alternates').is_file()
            assert _git(source.root, 'rev-parse', 'HEAD') == source.head
            assert _git(source.root, 'status', '--porcelain') == ''

        assert not root.exists()
        assert list(scratch.iterdir()) == []

    async def test_the_clone_is_removed_when_the_block_raises(
        self, source: _Source, scratch: Path,
    ):
        diff = _fixture(
            _CHANGE_AT_BASE, cwd=source.root, base_sha=source.base,
            changed_files=['a.py', 'new.py'],
        )
        seen: list[Path] = []

        with pytest.raises(RuntimeError, match='review blew up'):
            async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
                seen.append(checkout.root)
                raise RuntimeError('review blew up')

        assert seen and not seen[0].exists()
        assert list(scratch.iterdir()) == []

    async def test_a_hunk_header_that_miscounts_its_lines_still_applies(
        self, source: _Source, scratch: Path,
    ):
        diff = _fixture(
            _MISCOUNTED_HUNK, cwd=source.root, base_sha=source.base,
            changed_files=['a.py'],
        )

        async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
            assert checkout.unapplied_reason is None
            assert checkout.change_base == source.base
            assert (checkout.root / 'a.py').read_text() == 'x = 10\ny = 20\n'


@pytest.mark.asyncio
class TestNotApplicableAtItsBase:
    @pytest.mark.parametrize(
        ('diff_text', 'stderr_names'),
        [
            pytest.param(_FICTIONAL_PRE_IMAGE, 'a.py', id='fictional-pre-image'),
            pytest.param(_REAL_WORLD_FORMAT, 'patch', id='real-world-format'),
        ],
    )
    async def test_head_stays_at_the_base_with_gits_reason(
        self, source: _Source, scratch: Path, diff_text: str, stderr_names: str,
    ):
        diff = _fixture(
            diff_text, cwd=source.root, base_sha=source.base, changed_files=['a.py'],
        )

        async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
            assert checkout.change_base is None
            assert _git(checkout.root, 'rev-parse', 'HEAD') == source.base
            assert checkout.unapplied_reason
            assert stderr_names in checkout.unapplied_reason
            assert (checkout.root / 'a.py').read_text() == 'x = 1\n'

        assert list(scratch.iterdir()) == []


@pytest.mark.asyncio
class TestUnknownBase:
    async def test_head_is_the_source_head_and_the_recorded_reason_is_carried(
        self, source: _Source, scratch: Path,
    ):
        diff = _fixture(
            _CHANGE_AT_BASE, cwd=source.root, base_sha=None, changed_files=['a.py'],
            base_sha_reason='synthetic diff never applied to any commit',
        )

        async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
            assert checkout.change_base is None
            assert _git(checkout.root, 'rev-parse', 'HEAD') == source.head
            assert checkout.unapplied_reason == 'synthetic diff never applied to any commit'


@pytest.mark.asyncio
class TestSubprojectFixture:
    async def test_a_subdirectory_cwd_checks_out_the_whole_repo_at_its_root(
        self, source: _Source, scratch: Path,
    ):
        diff = _fixture(
            _GUI_CHANGE, cwd=source.root / 'gui', base_sha=source.base,
            changed_files=['gui/g.py'],
        )

        async with fixture_checkout(diff, scratch_parent=scratch) as checkout:
            toplevel = _git(checkout.root, 'rev-parse', '--show-toplevel')
            assert Path(toplevel).resolve() == checkout.root.resolve()
            assert checkout.change_base == source.base
            assert (checkout.root / 'gui' / 'g.py').read_text() == 'g = 2\n'
            assert (checkout.root / 'a.py').is_file()
