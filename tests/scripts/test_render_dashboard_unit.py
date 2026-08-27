"""Tests for scripts/render_dashboard_unit.py and the ``environment_map`` lift.

WHAT THE RENDERER EXISTS TO PREVENT. setup-host.sh used to install the
dashboard unit with a plain truncating redirect::

    sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" ... > "$UNIT_DIR/dark-factory-dashboard.service"

scripts/dashboard.service.template declares
``Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=__REPO_ROOT__``, so that render
collapsed this host's NINE measured aggregation roots to one — and did it
INVISIBLY, because DASHBOARD_KNOWN_PROJECT_ROOTS is on
check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST (compared by variable NAME,
value blessed), so the post-install parity check reported green afterwards.

ALL FIXTURES ARE tmp_path OR IN-MEMORY STRINGS — NEVER ~/.config/systemd/user/.
The same rule tests/scripts/test_check_dashboard_unit_parity.py,
test_check_fused_memory_unit_parity.py and test_dashboard_installed_unit_parity.py
each state in their own docstrings, and it is load-bearing here for the same
reason: the installed dashboard unit on this host is deliberately stale (task
4445), so any assertion made against it would encode host state rather than
renderer behaviour. The only real-tree reads are REPO-side (the committed
scripts/dashboard.service.template and dashboard/dark-factory-dashboard.service).

Module loading: scripts/ is not a package, so the checker is loaded via
importlib.util.spec_from_file_location, mirroring
tests/scripts/test_check_dashboard_unit_parity.py::_load_checker.
``systemd_unit_parity`` and ``render_dashboard_unit`` are imported by NAME —
tests/scripts/conftest.py inserts scripts/ onto sys.path for exactly this
(pyproject's ``--import-mode=importlib`` deliberately does not).
"""

import importlib.util
import pathlib
import types

REPO_ROOT = pathlib.Path(__file__).parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_dashboard_unit_parity.py"
TEMPLATE_PATH = REPO_ROOT / "scripts" / "dashboard.service.template"
HARDCODED_PATH = REPO_ROOT / "dashboard" / "dark-factory-dashboard.service"


def _load_checker() -> types.ModuleType:
    """Load scripts/check_dashboard_unit_parity.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "check_dashboard_unit_parity", CHECKER_PATH
    )
    assert spec is not None, f"Could not build spec from {CHECKER_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The `environment_map` lift  (step-1 / step-2)
# ---------------------------------------------------------------------------
# The renderer needs the shlex-correct Environment= reader the parity checker
# already has, and must NOT get it by importing the checker: the section-8
# harness in tests/scripts/test_check_dashboard_unit_parity.py builds a tmp repo
# in which `write_checker(body=...)` replaces check_dashboard_unit_parity.py
# with an argparse-usage-error STUB (and, with with_checker=False, omits it
# entirely). A renderer importing that module would ImportError under exactly
# the two tests asserting the install still happens when the gate did not run.
# So the shared dependency goes DOWN into scripts/systemd_unit_parity.py.


def test_environment_map_lives_on_the_shared_module():
    """``environment_map`` is importable from the SHARED module, by name.

    The renderer reaches it here, never through the checker — see the section
    comment above for the concrete obstacle (a stubbed checker in the
    section-8 tmp repo) that makes the import direction load-bearing rather
    than a matter of taste.
    """
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

    assert callable(systemd_unit_parity.environment_map)


def test_dashboard_checker_consumes_the_lifted_environment_map():
    """IDENTITY guard for the THIRD lift: ``environment_map`` has one home.

    Same assertion shape, and the same reason, as the parser and find_dropins
    guards in tests/scripts/test_check_orchestrator_unit_parity.py: re-exporting
    keeps the checker's module surface intact (its suite reads
    ``mod._environment_map`` directly), but a re-export is only meaningful if it
    is the SAME function object. Paste a look-alike copy back into the checker
    and every other test in both suites stays green while the two
    implementations quietly drift — inside the tooling built to catch exactly
    that. Object identity is the only check that fires on it.
    """
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

    checker = _load_checker()

    assert checker._environment_map is systemd_unit_parity.environment_map


# The parsing behaviour that must survive the move, driven straight off the
# docstring the lift carries with it. Exercised against the SHARED module, so
# the behaviour is pinned at its new home rather than through a re-export.


def _env(text, section="Service"):
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

    return systemd_unit_parity.environment_map(
        systemd_unit_parity.parse_unit_directives(text), section
    )


def test_environment_map_reads_several_assignments_on_one_line():
    """``Environment=A=1 B=2`` is TWO variables, the way systemd reads it.

    Before this used shlex it parsed as the single variable ``A`` with value
    ``1 B=2`` — inventing drift out of a pure reformat against a copy using the
    one-per-line spelling.
    """
    assert _env("[Service]\nEnvironment=A=1 B=2\n") == {"A": "1", "B": "2"}


def test_environment_map_reads_quoted_assignments():
    """``Environment="A=1" "B=2"`` yields the same map as the bare spelling.

    Before shlex, ``Environment="A=1"`` produced a variable literally named
    ``"A``.
    """
    assert _env('[Service]\nEnvironment="A=1" "B=2"\n') == {"A": "1", "B": "2"}


def test_environment_map_reads_the_one_per_line_spelling():
    """The third spelling of the same thing. All three must agree."""
    assert _env("[Service]\nEnvironment=A=1\nEnvironment=B=2\n") == {
        "A": "1",
        "B": "2",
    }


def test_environment_map_splits_on_the_first_equals_only():
    """``A=b=c`` sets A to ``b=c`` — the value may itself contain ``=``."""
    assert _env("[Service]\nEnvironment=A=b=c\n") == {"A": "b=c"}


def test_environment_map_last_occurrence_wins():
    """systemd applies Environment= directives in FILE ORDER, so the later one wins.

    A reader that kept the first would report the value the running unit does
    not have.
    """
    assert _env("[Service]\nEnvironment=A=1\nEnvironment=A=2\n") == {"A": "2"}


def test_environment_map_skips_a_token_with_no_equals():
    """A token carrying no ``=`` is skipped rather than guessed at."""
    assert _env("[Service]\nEnvironment=BARE A=1\n") == {"A": "1"}


def test_environment_map_falls_back_to_the_whole_line_on_unbalanced_quotes():
    """A malformed value must show up as a VARIABLE, not vanish into silent parity.

    systemd would reject the line too, but a reader that dropped it would let a
    broken installed unit compare equal to a correct one.
    """
    assert _env('[Service]\nEnvironment=A="1\n') == {"A": '"1'}


def test_environment_map_returns_empty_for_an_absent_section():
    """No [Service] section at all yields {} rather than raising."""
    assert _env("[Unit]\nDescription=x\n") == {}


# ---------------------------------------------------------------------------
# render_template  (step-3 / step-4)
# ---------------------------------------------------------------------------
# The substitution half: exactly what setup-host.sh's two `sed -e` expressions
# did, now owned by one function so the installer, the byte-for-byte
# template/committed-unit lockstep guard in
# tests/scripts/test_dashboard_service_template.py, and this suite all name the
# SAME spelling of it instead of three replayed copies.

# The literal paths baked into dashboard/dark-factory-dashboard.service, and
# the values the committed template must expand to in order to produce it.
HARDCODED_SERVICE_REPO_ROOT = "/home/leo/src/dark-factory"
HARDCODED_SERVICE_UV_PATH = "/home/leo/.local/bin/uv"


def _render_template(template_text, **kwargs):
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.render_template(template_text, **kwargs)


def test_render_template_reproduces_the_committed_unit_byte_for_byte():
    """The canonical lockstep claim, made THROUGH the code the installer runs.

    Rendering the committed template at the hardcoded host's REPO_ROOT/UV_PATH
    must yield dashboard/dark-factory-dashboard.service exactly. Asserted here
    as well as in test_dashboard_service_template.py because after this task
    setup-host.sh no longer carries a `sed` for this unit — this function IS
    the substitution, so its own suite has to pin the property rather than
    inheriting it.
    """
    rendered = _render_template(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        repo_root=HARDCODED_SERVICE_REPO_ROOT,
        uv_path=HARDCODED_SERVICE_UV_PATH,
    )

    assert rendered == HARDCODED_PATH.read_text(encoding="utf-8"), (
        f"Rendered template does not match {HARDCODED_PATH}. The template and "
        "the committed unit have drifted; re-render with render_template and "
        "update dashboard/dark-factory-dashboard.service."
    )


def test_render_template_leaves_no_sentinel_behind():
    """Neither sentinel may survive the render.

    An unsubstituted ``__REPO_ROOT__`` in WorkingDirectory= is not a cosmetic
    leftover: systemd rejects the unit outright with "WorkingDirectory= path is
    not absolute".
    """
    rendered = _render_template(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        repo_root="/srv/df",
        uv_path="/opt/uv",
    )

    assert "__REPO_ROOT__" not in rendered, rendered
    assert "__UV_PATH__" not in rendered, rendered


def test_render_template_substitutes_every_occurrence():
    """GLOBAL, like `sed s|...|...|g` — not just the first match.

    The committed template carries __REPO_ROOT__ several times (Documentation=,
    WorkingDirectory=, and two Environment= values at least). A first-match-only
    substitution would leave the rest as sentinels, which the sentinel guard
    above catches — this one pins the COUNT, so a render that substituted all
    but changed how many sites exist is still visible.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    occurrences = template.count("__REPO_ROOT__")
    assert occurrences >= 4, (
        f"Expected the committed template to carry __REPO_ROOT__ at several "
        f"sites; found {occurrences}. If the template genuinely shrank, this "
        "guard needs re-deriving — it exists so a first-match-only "
        "substitution cannot pass."
    )

    rendered = _render_template(template, repo_root="/srv/df", uv_path="/opt/uv")

    assert rendered.count("/srv/df") == occurrences
    assert rendered.count("/opt/uv") == template.count("__UV_PATH__")


def test_render_template_substitutes_literally_not_as_a_regex():
    """A value carrying `|` or a regex metacharacter is inserted VERBATIM.

    The shell spelling this replaces was ``sed -e "s|__REPO_ROOT__|$REPO_ROOT|g"``,
    whose delimiter is ``|`` — a repo root containing one would have ended the
    expression and made sed fail (or, worse, silently mean something else).
    Pinning literal substitution here is what lets the shell form be retired
    without quietly changing the contract on an exotic path.
    """
    template = "[Service]\nWorkingDirectory=__REPO_ROOT__\nExecStart=__UV_PATH__ run\n"
    weird_root = "/srv/a|b/c.d/e*f/g[h]"
    weird_uv = "/opt/u|v/uv"

    rendered = _render_template(template, repo_root=weird_root, uv_path=weird_uv)

    assert rendered == (
        f"[Service]\nWorkingDirectory={weird_root}\nExecStart={weird_uv} run\n"
    )
