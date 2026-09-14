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

import pytest

from fused_memory.config.schema import FusedMemoryConfig

#: A value nothing in the tracked config or the schema could ever produce, so
#: seeing it back out of ``FusedMemoryConfig()`` can only mean the environment
#: reached in.
PWNED = '/pwned-by-env'


def test_config_resolution_is_independent_of_the_process_cwd(monkeypatch, tmp_path):
    """The same commit must resolve the same config from any CWD.

    `.taskmaster` is the sharpest single probe: its code default is `None`
    while the tracked `fused-memory/config/config.yaml` carries a real
    `taskmaster:` section, so it flips precisely when the YAML layer is lost.
    The full `model_dump()` comparison then catches every other field that
    would silently drop to a code default alongside it.
    """
    before_chdir = FusedMemoryConfig()

    monkeypatch.chdir(tmp_path)
    after_chdir = FusedMemoryConfig()

    assert before_chdir.taskmaster is not None
    assert after_chdir.taskmaster is not None
    assert after_chdir.model_dump() == before_chdir.model_dump()


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
    environment variable named after any of the model's nineteen top-level
    fields is an unprefixed override — and env settings outrank the YAML.
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
