"""The suite's config resolution must not depend on where pytest was launched.

Task 5444.  `FusedMemoryConfig` is a pydantic-settings `BaseSettings`, not a
plain `BaseModel`: `fused_memory.config.schema::FusedMemoryConfig.settings_customise_sources`
resolves `CONFIG_PATH`, defaulting to the RELATIVE `config/config.yaml`, and
`fused_memory.config.schema::YamlSettingsSource.__call__` silently returns `{}`
when that path does not exist.  So every field a caller does not pass
explicitly is filled from a file found relative to the process CWD — and from
the ambient environment, since the model sets `env_prefix=''`.

That made a test's outcome a function of the pytest CWD.  Measured on main tip
bd4574c8d1, on the same commit, with the same test:

    uv run --directory fused-memory pytest tests/test_referent_repair.py \
        -k taskmaster_project_root_is_never_used          -> 1 passed
    uv run --project   fused-memory pytest fused-memory/tests/test_referent_repair.py \
        -k taskmaster_project_root_is_never_used          -> 1 failed

Green for the gate (which runs pytest from `fused-memory/`), red for anyone
running it from the repo root.  The tests here pin that invariant directly so
the leak cannot silently return; `conftest.py::_isolate_fm_config` is what
holds it.
"""

import json
import os
from pathlib import Path

import pytest

from fused_memory.config.schema import FusedMemoryConfig

#: A value nothing in the tracked config or the schema could ever produce, so
#: seeing it back out of ``FusedMemoryConfig()`` can only mean the environment
#: reached in.
PWNED = '/pwned-by-env'

#: Marks a variable these tests planted themselves, for the near-miss class below.
SURVIVES = 'kept-by-the-narrow-scrub'

#: Names the scrub must LEAVE ALONE.  The first group are near-misses on its
#: two rules — ``PATH`` is not ``path_scope_adjudicator``; ``MEM0_DIR`` and
#: ``MEM0_TELEMETRY`` are not ``mem0``; ``TASKMASTER_PROJECT_ROOT`` spells the
#: nesting with ONE underscore, not the ``__`` delimiter; ``FOO__TASKMASTER``
#: has the head ``foo``.  The rest are ambient variables the other ~19.8k
#: tests, the uv/venv machinery and the real-I/O cohort need alive.
NEAR_MISS_ENV_NAMES = (
    'PATH',
    'HOME',
    'VIRTUAL_ENV',
    'OPENAI_API_KEY',
    'MEM0_API_KEY',
    'MEM0_DIR',
    'MEM0_TELEMETRY',
    'TASKMASTER_PROJECT_ROOT',
    'FOO__TASKMASTER',
)


def test_config_resolution_is_independent_of_the_process_cwd(monkeypatch, tmp_path):
    """The same commit must resolve the same config LAYERS from any CWD.

    `.taskmaster` is the sharpest single probe: its code default is `None`
    while the tracked `fused-memory/config/config.yaml` carries a real
    `taskmaster:` section, so it flips precisely when the YAML layer is lost.
    The full `model_dump()` comparison then catches every other field that
    would silently drop to a code default alongside it.

    READ THE EQUALITY AS WRITTEN: equal dumps are STRING equality, not
    semantic equality.  The tracked config's path leaves are relative, so
    `taskmaster.project_root` is the literal `'.'` on both sides of the chdir
    — equal, while denoting two different directories.  That residual is real
    and is NOT closed here; the next test states it outright rather than
    leaving this one to be misread as covering it.
    """
    before_chdir = FusedMemoryConfig()

    monkeypatch.chdir(tmp_path)
    after_chdir = FusedMemoryConfig()

    assert before_chdir.taskmaster is not None
    assert after_chdir.taskmaster is not None
    assert after_chdir.model_dump() == before_chdir.model_dump()


def test_the_pin_leaves_relative_path_leaves_denoting_the_launching_cwd(monkeypatch, tmp_path):
    """The accepted residual, stated executably: same string, different directory.

    `conftest.py::_isolate_fm_config` pins the config FILE by absolute path,
    which is what makes resolution CWD-independent — but the file it pins
    carries `${PROJECT_ROOT:.}` and `${QUEUE_DATA_DIR:./data/queue}`, so a
    test that RESOLVES one of those leaves still gets a directory that depends
    on where pytest was launched.  The orchestrator's fixture closes this half
    by pinning `ORCH_PROJECT_ROOT` at `tmp_path`; doing the same here would
    change the value every currently-green test reads, so it is recorded in
    `plans/fused-memory-config-cwd-leak-rca-2026-09-13.md` and filed as a
    follow-up rather than done under this task's zero-collateral decision.

    This test exists so the residual cannot be mistaken for fixed.  When the
    follow-up lands, the second assertion is the one meant to fail: invert it
    rather than deleting it.

    `.resolve()` on the before-value is taken BEFORE the chdir on purpose — a
    relative path resolves against the CWD *at the moment it is resolved*, so
    resolving both afterwards compares the new CWD with itself and the test
    passes vacuously whatever the fixture does.  It did exactly that on the
    first draft.
    """
    before_chdir = FusedMemoryConfig().taskmaster
    assert before_chdir is not None
    before_resolved = Path(before_chdir.project_root).resolve()

    monkeypatch.chdir(tmp_path)
    after_chdir = FusedMemoryConfig().taskmaster
    assert after_chdir is not None

    assert after_chdir.project_root == before_chdir.project_root
    assert Path(after_chdir.project_root).resolve() != before_resolved


class TestAmbientInterpolationCannotRedirectTheConfig:
    """The half of the leaf story that IS closed: the inherited redirect.

    ``${PROJECT_ROOT:.}``, ``${QUEUE_DATA_DIR:...}`` and
    ``${RECONCILIATION_DATA_DIR:...}`` are interpolated by the YAML loader, not
    by pydantic-settings, so they are a THIRD env surface: the scrub derived
    from ``model_fields`` cannot see them and the file pin does not outrank
    them.  Measured before ``_isolate_fm_config`` listed them — with
    ``CONFIG_PATH`` already pinned at the canonical file,
    ``PROJECT_ROOT=/pwned-by-env`` resolved both ``taskmaster.project_root``
    and ``reconciliation.explore_codebase_root`` to ``/pwned-by-env``.
    Inheriting one is a real launch condition, not a contrived one: this
    repo's own operator scripts export ``PROJECT_ROOT``.

    Planting AMBIENTLY is the whole point, exactly as in
    ``TestAmbientEnvCannotRewriteTheConfig``.  A value set in a test BODY still
    reaches the config, because the interpolation reads ``os.environ`` when
    ``FusedMemoryConfig()`` is constructed — the fixture removes what pytest
    INHERITED and deliberately leaves a test's own override working.  An
    earlier draft of this asserted the opposite and failed, which is the
    sharper statement of the boundary than any docstring.
    """

    @pytest.fixture(
        scope='class',
        autouse=True,
        params=[
            ('PROJECT_ROOT', lambda config: config.taskmaster.project_root),
            ('QUEUE_DATA_DIR', lambda config: config.queue.data_dir),
            ('RECONCILIATION_DATA_DIR', lambda config: config.reconciliation.data_dir),
        ],
        ids=['project-root', 'queue-data-dir', 'reconciliation-data-dir'],
    )
    def redirected_leaf(self, request):
        """Plant one interpolation variable ambiently; hand back its leaf reader.

        The reader travels WITH the name as a callable rather than as a dotted
        string the test would have to split: the pairing is the point of the
        case, and a parsed string would be one more thing that can drift from
        the config it names.
        """
        name, read_leaf = request.param
        ambient = pytest.MonkeyPatch()
        ambient.setenv(name, PWNED)
        yield read_leaf
        ambient.undo()

    def test_an_inherited_interpolation_var_cannot_redirect_the_config(self, redirected_leaf):
        """The planted directory must not reach the leaf it interpolates into."""
        config = FusedMemoryConfig()

        assert config.taskmaster is not None
        assert redirected_leaf(config) != PWNED


def test_code_default_config_yields_pure_code_defaults(code_default_config):
    """The escape hatch the pin makes necessary: schema defaults, on request.

    With `CONFIG_PATH` now always naming a real file, a test that wants to
    assert what the SCHEMA declares — rather than what the tracked
    `fused-memory/config/config.yaml` overrides it to — has no other way to
    get there.  `.taskmaster` is the sharpest probe for the same reason as
    above: its code default is `None` and the YAML supplies a real section, so
    it reads `None` exactly when the YAML layer is genuinely absent.
    """
    assert FusedMemoryConfig().taskmaster is None


class TestAmbientEnvCannotRewriteTheConfig:
    """The other half of the same leak, and the one the file pin does NOT close.

    ``FusedMemoryConfig`` sets ``env_prefix=''`` with
    ``env_nested_delimiter='__'`` and ``case_sensitive=False``, so a BARE
    environment variable named after any of the model's top-level fields is an
    unprefixed override — and env settings outrank the YAML.
    Whatever shell, CI runner or parent process happens to export
    ``TASKMASTER`` can therefore rewrite a value a test reads.  Measured: this
    reproduces with ``CONFIG_PATH`` pointing at a missing file, which is what
    makes it independent of the CWD half.
    """

    @pytest.fixture(
        scope='class',
        autouse=True,
        params=[
            ('TASKMASTER', json.dumps({'project_root': PWNED})),
            ('TASKMASTER__PROJECT_ROOT', PWNED),
        ],
        ids=['bare-field-name', 'nested-delimiter'],
    )
    def _ambient_taskmaster(self, request):
        """Plant the hostile variable AMBIENTLY — before per-test isolation runs.

        Class scope is load-bearing, not tidiness: pytest instantiates
        higher-scoped fixtures first, so this runs before the function-scoped
        autouse ``conftest.py::_isolate_fm_config`` and the variable is
        already there when isolation happens.  That is precisely the shape of
        the real defect — a variable inherited from whoever launched pytest.
        Setting it in the test BODY instead would model nothing: it would land
        after isolation, and a deliberate test-local override is a documented
        escape hatch rather than a leak.
        """
        name, value = request.param
        ambient = pytest.MonkeyPatch()
        ambient.setenv(name, value)
        yield name
        ambient.undo()

    def test_an_ambient_bare_env_var_cannot_rewrite_the_suite_config(self):
        """The planted variable must not reach the value a test would read.

        `.taskmaster` is `TaskmasterConfig | None`, and the scrub leaves the
        tracked YAML's section in place, so pinning it non-None first reports
        a lost YAML layer as itself rather than as an attribute error on the
        assertion that matters.
        """
        taskmaster = FusedMemoryConfig().taskmaster

        assert taskmaster is not None
        assert taskmaster.project_root != PWNED


class TestTheScrubStaysNarrow:
    """The scrub's NARROWNESS is as load-bearing as its coverage.

    ``conftest.py::_isolate_fm_config`` deletes inherited variables from the
    whole suite's environment, so a widened match — someone reaching for a
    prefix or a substring test — would delete ``PATH`` and ``OPENAI_API_KEY``
    for all ~19.8k tests.  That regression surfaces as a diffuse cascade of
    unrelated failures in whichever suite shells out or talks to a backend
    first, nowhere near the conftest that caused it.  Until this class existed
    the narrowness was asserted only in prose, in the very docstring a
    widening edit would be rewriting.

    The other direction is already covered: ``TestAmbientEnvCannotRewriteTheConfig``
    above fails if the scrub is removed.
    """

    @pytest.fixture(scope='class', autouse=True)
    def ambient_near_misses(self):
        """Plant the near-miss names AMBIENTLY and hand back what each holds.

        Class scope for the same reason as ``_ambient_taskmaster`` above:
        higher-scoped fixtures are set up first, so these names are already in
        the environment when the function-scoped autouse isolation runs — the
        shape of a variable inherited from whoever launched pytest.

        An EXISTING value is kept rather than overwritten.  ``PATH`` and
        ``HOME`` are on this list precisely because things break without them,
        and a sentinel ``PATH`` held for the length of a class would be a
        worse hazard than the regression being guarded against; a name that is
        absent gets a sentinel so the assertion still observes a plant rather
        than an absence.
        """
        ambient = pytest.MonkeyPatch()
        for name in NEAR_MISS_ENV_NAMES:
            if name not in os.environ:
                ambient.setenv(name, f'{SURVIVES}-{name}')
        planted = {name: os.environ[name] for name in NEAR_MISS_ENV_NAMES}
        yield planted
        ambient.undo()

    @pytest.mark.parametrize('name', NEAR_MISS_ENV_NAMES)
    def test_a_near_miss_name_survives_the_scrub(self, name, ambient_near_misses):
        """Each name is still readable, with its value intact, inside a test.

        Parametrised one name per case so a widening reports WHICH rule was
        loosened — a single test asserting all nine would name only the first.
        """
        assert os.environ.get(name) == ambient_near_misses[name]
