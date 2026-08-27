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

import pytest

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


# ---------------------------------------------------------------------------
# preserved_values / apply_preserved  (step-5 / step-6)
# ---------------------------------------------------------------------------
# The preservation MECHANISM, exercised with an EXPLICIT `names` argument — no
# module constant is consulted here. Which names are host-local is POLICY, and
# it is pinned separately (see HOST_LOCAL_ENVIRONMENT below), so that a policy
# change and a mechanism regression cannot be mistaken for one another.

_KNOWN_ROOTS = "DASHBOARD_KNOWN_PROJECT_ROOTS"

# The measured value on this host, recorded in
# tests/scripts/test_dashboard_installed_unit_parity.py's docstring: NINE
# project roots where the committed template pins one. This is the exact data
# the old `sed >` render destroyed on every re-run.
NINE_ROOTS = ",".join(
    f"/home/leo/src/{name}"
    for name in (
        "dark-factory",
        "reify",
        "autopilot-video",
        "autotrade",
        "know-live",
        "solar-challenge",
        "mission-control",
        "solar-challenge-platform",
        "pump-web-ui",
    )
)


def _preserved_values(installed_text, names):
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.preserved_values(installed_text, names)


def _apply_preserved(rendered_text, preserved):
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.apply_preserved(rendered_text, preserved)


def _rendered(repo_root="/home/leo/src/dark-factory"):
    """The committed template rendered at *repo_root* — the single-root default."""
    return _render_template(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        repo_root=repo_root,
        uv_path=HARDCODED_SERVICE_UV_PATH,
    )


def _installed_with_nine_roots(repo_root="/home/leo/src/dark-factory"):
    """A realistic INSTALLED unit: the rendered template, nine roots substituted in.

    Built by rendering rather than hand-written, so the fixture cannot drift
    into a shape the real installer would never produce.
    """
    text = _rendered(repo_root)
    single = f"Environment={_KNOWN_ROOTS}={repo_root}"
    assert single in text, f"fixture anchor {single!r} not found in the rendered template"
    return text.replace(single, f"Environment={_KNOWN_ROOTS}={NINE_ROOTS}")


def test_preserved_values_reads_the_host_local_value_off_the_installed_unit():
    """(a) The nine roots are found on a realistically-shaped installed unit."""
    preserved, skipped = _preserved_values(_installed_with_nine_roots(), (_KNOWN_ROOTS,))

    assert preserved == {_KNOWN_ROOTS: NINE_ROOTS}
    assert skipped == {}


def test_apply_preserved_rewrites_only_the_named_line():
    """(b) The preserved value goes back, and NOTHING else moves.

    Asserted line-by-line rather than by a single equality, so a failure names
    the line that changed instead of printing two 119-line units.
    """
    rendered = _rendered()
    result = _apply_preserved(rendered, {_KNOWN_ROOTS: NINE_ROOTS})

    before = rendered.splitlines()
    after = result.splitlines()
    assert len(before) == len(after), "apply_preserved changed the line COUNT"

    changed = [i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a]
    assert len(changed) == 1, (
        f"Expected exactly one changed line; {len(changed)} changed: "
        f"{[after[i] for i in changed]}"
    )
    assert after[changed[0]] == f"Environment={_KNOWN_ROOTS}={NINE_ROOTS}"


def test_preserved_values_greenfield_preserves_nothing():
    """(c) No installed unit at all → the rendered single-root default survives.

    A bare host is the case setup-host.sh exists to serve. Preserving nothing
    here must be silent-of-error but NOT silent-of-record: the name is reported
    as skipped, so the install still leaves a trace of what it decided.
    """
    preserved, skipped = _preserved_values("", (_KNOWN_ROOTS,))

    assert preserved == {}
    assert _KNOWN_ROOTS in skipped
    assert "absent" in skipped[_KNOWN_ROOTS].lower(), skipped

    rendered = _rendered()
    assert _apply_preserved(rendered, preserved) == rendered


def test_preserved_values_ignores_a_unit_that_never_declares_the_name():
    """(c, second shape) An installed unit that simply lacks the variable."""
    installed = "[Service]\nType=simple\nEnvironment=SOMETHING_ELSE=1\n"

    preserved, skipped = _preserved_values(installed, (_KNOWN_ROOTS,))

    assert preserved == {}
    assert "absent" in skipped[_KNOWN_ROOTS].lower(), skipped


def test_preserved_values_refuses_an_empty_installed_value():
    """(d) An EMPTY installed value is not preserved, and the fallback is REPORTED.

    An empty DASHBOARD_KNOWN_PROJECT_ROOTS is not a usable aggregation scope —
    preserving it would be strictly worse than the rendered single-root default,
    which at least aggregates this repo. But taking the default SILENTLY is the
    failure this whole change is about, so the skip has to appear in the record.
    """
    installed = f"[Service]\nEnvironment={_KNOWN_ROOTS}=\n"

    preserved, skipped = _preserved_values(installed, (_KNOWN_ROOTS,))

    assert preserved == {}
    assert "empty" in skipped[_KNOWN_ROOTS].lower(), skipped


def test_preserved_values_refuses_a_whitespace_only_installed_value():
    """(d, second shape) Whitespace-only is empty for this purpose."""
    installed = f'[Service]\nEnvironment="{_KNOWN_ROOTS}=   "\n'

    preserved, skipped = _preserved_values(installed, (_KNOWN_ROOTS,))

    assert preserved == {}
    assert "empty" in skipped[_KNOWN_ROOTS].lower(), skipped


def test_preserved_values_reads_the_multi_assignment_spelling():
    """(e) `Environment=FOO=1 KNOWN=/a,/b` is read correctly.

    This is exactly what the step-2 lift buys: the shlex-correct reader the
    parity checker compares with. A naive split would have yielded the single
    variable FOO with value `1 DASHBOARD_KNOWN_PROJECT_ROOTS=/a,/b`, preserved
    nothing, and reported success.
    """
    installed = f"[Service]\nEnvironment=FOO=1 {_KNOWN_ROOTS}=/a,/b\n"

    preserved, skipped = _preserved_values(installed, (_KNOWN_ROOTS,))

    assert preserved == {_KNOWN_ROOTS: "/a,/b"}
    assert skipped == {}


def test_apply_preserved_raises_when_the_rendered_text_has_no_such_line():
    """(f) A preservation that would silently no-op RAISES, naming the variable.

    Zero matching lines means the template changed shape underneath this code:
    the value would be read off the installed unit, dropped on the floor, and
    the install would report success — the exact silent clobber this whole
    change removes, one layer in. It must be loud.
    """
    rendered = "[Service]\nType=simple\n"

    with pytest.raises(ValueError) as excinfo:
        _apply_preserved(rendered, {_KNOWN_ROOTS: NINE_ROOTS})

    assert _KNOWN_ROOTS in str(excinfo.value), excinfo.value


def test_apply_preserved_raises_when_the_rendered_text_is_ambiguous():
    """(f, second shape) More than one matching line is ambiguous — also loud.

    systemd's last-wins would make the choice invisible, so rewriting one of
    two and leaving the other is a value nobody chose.
    """
    rendered = (
        f"[Service]\nEnvironment={_KNOWN_ROOTS}=/a\nEnvironment={_KNOWN_ROOTS}=/b\n"
    )

    with pytest.raises(ValueError) as excinfo:
        _apply_preserved(rendered, {_KNOWN_ROOTS: NINE_ROOTS})

    assert _KNOWN_ROOTS in str(excinfo.value), excinfo.value


# ---------------------------------------------------------------------------
# HOST_LOCAL_ENVIRONMENT — the POLICY  (step-7 / step-8)
# ---------------------------------------------------------------------------


def _host_local():
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.HOST_LOCAL_ENVIRONMENT


def test_host_local_environment_is_not_empty():
    """An empty preserve set would make this whole module a rename of the clobber."""
    assert _host_local(), (
        "HOST_LOCAL_ENVIRONMENT is empty, so the renderer preserves nothing and "
        "the install is byte-for-byte the `sed >` clobber it replaced — while "
        "reporting success."
    )


def test_host_local_environment_is_a_subset_of_the_divergence_allowlist():
    """(a) Preserving a name the checker VALUE-COMPARES would turn the gate red forever.

    check_dashboard_unit_parity compares Environment= values across the
    committed and installed copies unless the name is on DIVERGENCE_ALLOWLIST.
    Preserve a name that is NOT on that list and every correctly-configured host
    reports drift on every run — the always-red-gate outcome that checker's own
    docstring warns about twice, and which a gate then gets switched off for,
    taking the accidental drift it exists to catch with it.

    Held by a guard rather than by an import, deliberately: the renderer must
    not import the checker (see the module docstring and the section comment on
    the environment_map lift above).
    """
    allowlist = set(_load_checker().DIVERGENCE_ALLOWLIST)

    assert set(_host_local()) <= allowlist, (
        f"HOST_LOCAL_ENVIRONMENT {sorted(_host_local())} is not a subset of "
        f"DIVERGENCE_ALLOWLIST {sorted(allowlist)}. A preserved name the parity "
        "checker value-compares makes the gate red on every correctly-"
        "configured host."
    )


def test_host_local_environment_contains_known_project_roots():
    """(b) The variable this task exists for."""
    assert _KNOWN_ROOTS in _host_local()


def test_host_local_environment_excludes_project_root():
    """(c) The two DIVERGENCE_ALLOWLIST entries are NOT interchangeable.

    They are on that allowlist for OPPOSITE reasons, and the naive reading
    ("preserve everything on the allowlist") destroys the second:

    - DASHBOARD_KNOWN_PROJECT_ROOTS is the declared HOLE — genuinely host-local,
      "additional project roots are LOCAL settings, added to the installed unit,
      not committed here". Preserving it is the whole point.
    - DASHBOARD_PROJECT_ROOT is NOT host-local. Its value is RENDERED from
      __REPO_ROOT__ per host, and the checker still CHECKS it — intra-copy,
      against the SAME file's WorkingDirectory= (UnitSpec.env_matches_directive).
      Preserving a PREVIOUS host's value would pin the dashboard's data root at
      the OLD repo root while WorkingDirectory= moved to the new one,
      manufacturing exactly the intra-copy drift _compare_env_matches_directive
      exists to report — on a host that had just been correctly reinstalled.
    """
    assert "DASHBOARD_PROJECT_ROOT" not in _host_local(), (
        "DASHBOARD_PROJECT_ROOT must NOT be preserved: its value is rendered "
        "per host and must equal the same copy's WorkingDirectory=. See this "
        "test's docstring."
    )


def test_host_local_environment_names_are_declared_in_the_committed_template():
    """(d) STALENESS GUARD: a typo must not rot into a preserve-nothing no-op.

    Mirrors test_divergence_allowlist_names_are_declared_in_a_committed_unit.
    A misspelled name here would be absent from every installed unit forever,
    so preserved_values would skip it on every host, the render would take the
    default every time, and the renderer would report success while doing
    exactly what the `sed >` it replaced did.
    """
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

    declared = set(
        systemd_unit_parity.environment_map(
            systemd_unit_parity.parse_unit_directives(
                TEMPLATE_PATH.read_text(encoding="utf-8")
            ),
            "Service",
        )
    )

    for name in _host_local():
        assert name in declared, (
            f"HOST_LOCAL_ENVIRONMENT names {name}, but "
            f"{TEMPLATE_PATH} declares no such Environment= variable. Declared: "
            f"{sorted(declared)}. A name nobody sets is preserved on no host, "
            "forever, while the renderer still reports success."
        )


# ---------------------------------------------------------------------------
# render_unit — the end-to-end composition  (step-9 / step-10)
# ---------------------------------------------------------------------------
# This is where ACCEPTANCE 3 is pinned, and it is pinned against the REAL,
# UNMODIFIED parity checker rather than by restating what parity means.

_OLD_ROOT = "/old/root"
_NEW_ROOT = "/srv/dark-factory"
_NEW_UV = "/opt/uv/bin/uv"


def _render_unit(**kwargs):
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.render_unit(
        TEMPLATE_PATH.read_text(encoding="utf-8"), **kwargs
    )


def _reinstall_over_a_configured_host():
    """Render at a NEW repo root over an installed unit rendered at an OLD one.

    The realistic re-provision: the checkout moved, and the host had nine
    aggregation roots configured into its installed unit.
    """
    return _render_unit(
        repo_root=_NEW_ROOT,
        uv_path=_NEW_UV,
        installed_text=_installed_with_nine_roots(_OLD_ROOT),
    )


def _env_line(text, name):
    prefix = f"Environment={name}="
    matches = [
        line.strip()[len(prefix) :]
        for line in text.splitlines()
        if line.strip().startswith(prefix)
    ]
    assert len(matches) == 1, f"expected one {prefix!r} line, found {len(matches)}"
    return matches[0]


def _directive(text, key):
    prefix = f"{key}="
    matches = [
        line.strip()[len(prefix) :]
        for line in text.splitlines()
        if line.strip().startswith(prefix)
    ]
    assert len(matches) == 1, f"expected one {prefix!r} line, found {len(matches)}"
    return matches[0]


def test_render_unit_preserves_the_nine_roots_across_a_moved_checkout():
    """(a) ACCEPTANCE 1: the host-local value SURVIVES the reinstall, verbatim.

    Nine roots in, nine roots out — asserted by COUNT as well as by equality, so
    a one-root result cannot pass by some partial match.
    """
    text, preserved, skipped = _reinstall_over_a_configured_host()

    assert _env_line(text, _KNOWN_ROOTS) == NINE_ROOTS
    assert _env_line(text, _KNOWN_ROOTS).count(",") == 8, "not nine roots"
    assert preserved == {_KNOWN_ROOTS: NINE_ROOTS}
    assert skipped == {}


def test_render_unit_re_derives_project_root_rather_than_preserving_it():
    """(b) The stale /old/root value must NOT survive — DASHBOARD_PROJECT_ROOT is rendered.

    This is the concrete failure the "preserve everything on the allowlist"
    reading would produce: the data root pinned at the previous checkout while
    WorkingDirectory= moved to the new one. The checker relates those two
    INSIDE one copy (UnitSpec.env_matches_directive), so that shape is drift
    the gate reports — manufactured by the installer, on a host that had just
    been correctly reinstalled.
    """
    text, _preserved, _skipped = _reinstall_over_a_configured_host()

    assert _OLD_ROOT not in text, (
        f"The previous repo root survived the re-render:\n{text}"
    )
    assert _env_line(text, "DASHBOARD_PROJECT_ROOT") == _NEW_ROOT
    assert _directive(text, "WorkingDirectory") == _NEW_ROOT
    assert _env_line(text, "DASHBOARD_PROJECT_ROOT") == _directive(
        text, "WorkingDirectory"
    )


def test_render_unit_output_is_at_parity_with_the_committed_unit():
    """(c) ACCEPTANCE 3: the REAL checker, unmodified, calls the result parity.

    compare_unit is the oracle rather than a restatement of what parity means:
    if preserving host-local values made the installed copy divergent by the
    gate's own reckoning, this is where it shows up. Note the render is at a
    DIFFERENT repo root and a DIFFERENT uv path from the committed copy, and
    carries nine roots where the committed copy carries one — all three are
    legitimate per-host divergences the spec already models, which is exactly
    what makes an empty result meaningful rather than trivial.
    """
    mod = _load_checker()
    text, _preserved, _skipped = _reinstall_over_a_configured_host()

    drifts = mod.compare_unit(
        mod.UNITS["dark-factory-dashboard.service"],
        HARDCODED_PATH.read_text(encoding="utf-8"),
        text,
    )

    assert drifts == [], f"The renderer's output is not at parity: {drifts}"


def test_compare_unit_would_have_reported_a_perturbed_value():
    """(d) ANTI-VACUITY: the empty result above is not an artefact of comparing nothing.

    Same guard-the-guard discipline the checker's own registry-staleness tests
    use. Perturb one compared, host-invariant literal and the oracle must fire.
    """
    mod = _load_checker()
    text, _preserved, _skipped = _reinstall_over_a_configured_host()
    perturbed = text.replace("TimeoutStopSec=15", "TimeoutStopSec=30")
    assert perturbed != text, "fixture perturbation target not found"

    drifts = mod.compare_unit(
        mod.UNITS["dark-factory-dashboard.service"],
        HARDCODED_PATH.read_text(encoding="utf-8"),
        perturbed,
    )

    assert drifts, "compare_unit reported parity on a changed TimeoutStopSec"
    assert any("TimeoutStopSec" in d.key for d in drifts), drifts


def test_compare_unit_still_reports_a_vanished_known_project_roots():
    """(d, second shape) Allowlisting a VALUE must never bless the variable VANISHING.

    The name-set branch of _compare_environment is what keeps the hole in the
    gate from becoming a hole for the whole variable — so a renderer that
    "preserved" DASHBOARD_KNOWN_PROJECT_ROOTS by deleting the line would still
    be caught. Pinned here because this suite's parity assertion above leans on
    that same allowlist.
    """
    mod = _load_checker()
    text, _preserved, _skipped = _reinstall_over_a_configured_host()
    dropped = "\n".join(
        line
        for line in text.splitlines()
        if not line.strip().startswith(f"Environment={_KNOWN_ROOTS}=")
    )
    assert f"Environment={_KNOWN_ROOTS}=" not in dropped

    drifts = mod.compare_unit(
        mod.UNITS["dark-factory-dashboard.service"],
        HARDCODED_PATH.read_text(encoding="utf-8"),
        dropped,
    )

    assert any(_KNOWN_ROOTS in d.key for d in drifts), (
        f"A vanished {_KNOWN_ROOTS} was not reported: {drifts}"
    )


def test_render_unit_greenfield_installs_the_single_root_default():
    """No installed unit → the rendered default, and the fallback is RECORDED."""
    text, preserved, skipped = _render_unit(
        repo_root=_NEW_ROOT, uv_path=_NEW_UV, installed_text=""
    )

    assert _env_line(text, _KNOWN_ROOTS) == _NEW_ROOT
    assert preserved == {}
    assert _KNOWN_ROOTS in skipped


def test_render_unit_defaults_its_preserve_set_to_the_policy_constant():
    """The default preserve set is HOST_LOCAL_ENVIRONMENT, not an empty tuple.

    A default of () would make every call site that omits `names` silently
    preserve nothing — the clobber back, opt-in.
    """
    explicit, _p, _s = _render_unit(
        repo_root=_NEW_ROOT,
        uv_path=_NEW_UV,
        installed_text=_installed_with_nine_roots(_OLD_ROOT),
        names=_host_local(),
    )
    defaulted, _p2, _s2 = _reinstall_over_a_configured_host()

    assert explicit == defaulted


# ---------------------------------------------------------------------------
# main(argv) — the CLI  (step-11 / step-12)
# ---------------------------------------------------------------------------
# All trees here are tmp_path. Nothing reads or writes ~/.config/systemd/user.

_TAG = "[dashboard_unit_render]"


def _main(argv):
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.main(argv)


def _cli(tmp_path, *, template_text=None, output_text=None, repo_root=_NEW_ROOT):
    """A tmp template (+ optional pre-existing --output) and the argv to render it."""
    template = tmp_path / "dashboard.service.template"
    template.write_text(
        TEMPLATE_PATH.read_text(encoding="utf-8")
        if template_text is None
        else template_text,
        encoding="utf-8",
    )
    output = tmp_path / "unit" / "dark-factory-dashboard.service"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_text is not None:
        output.write_text(output_text, encoding="utf-8")
    argv = [
        "--template", str(template),
        "--repo-root", repo_root,
        "--uv-path", _NEW_UV,
        "--output", str(output),
    ]
    return argv, output


def test_main_renders_to_output_and_returns_zero(tmp_path, capsys):
    """(a) The happy path: --output is written, exit 0."""
    argv, output = _cli(tmp_path)

    assert _main(argv) == 0
    text = output.read_text(encoding="utf-8")
    assert "__REPO_ROOT__" not in text and "__UV_PATH__" not in text
    assert _directive(text, "WorkingDirectory") == _NEW_ROOT
    capsys.readouterr()


def test_main_preserves_the_installed_value_it_reads_from_output(tmp_path, capsys):
    """(b) SELF-READ — and this is WHY the renderer owns --output.

    The obvious shape, `python3 render.py ... > "$UNIT_DIR/<unit>"`, is silently
    fatal: bash TRUNCATES a redirect target BEFORE the command runs, so the
    installed value would be gone before python could open the file. The tool
    would find nothing to preserve, take the rendered default, and REPORT
    SUCCESS — the same clobber one level up, now invisible because the thing
    that did it said it had preserved. Owning --output makes the read-then-write
    ordering structural rather than the caller's responsibility.
    """
    argv, output = _cli(
        tmp_path, output_text=_installed_with_nine_roots(_OLD_ROOT)
    )

    assert _main(argv) == 0
    assert _env_line(output.read_text(encoding="utf-8"), _KNOWN_ROOTS) == NINE_ROOTS
    capsys.readouterr()


def test_main_is_idempotent(tmp_path, capsys):
    """(b, continued) Running it twice in a row still leaves nine roots.

    Not a restatement of the test above: the SECOND run reads a file this tool
    itself wrote, so a renderer that preserved correctly only from a
    hand-configured unit — and not from its own output — would pass that one
    and fail this.
    """
    argv, output = _cli(
        tmp_path, output_text=_installed_with_nine_roots(_OLD_ROOT)
    )

    assert _main(argv) == 0
    first = output.read_text(encoding="utf-8")
    assert _main(argv) == 0
    second = output.read_text(encoding="utf-8")

    assert first == second
    assert _env_line(second, _KNOWN_ROOTS) == NINE_ROOTS
    capsys.readouterr()


def test_main_greenfield_writes_the_single_root_default(tmp_path, capsys):
    """(d) No pre-existing --output → the rendered default, exit 0."""
    argv, output = _cli(tmp_path)

    assert _main(argv) == 0
    assert _env_line(output.read_text(encoding="utf-8"), _KNOWN_ROOTS) == _NEW_ROOT
    capsys.readouterr()


def test_main_leaves_output_byte_unchanged_when_the_template_is_unreadable(
    tmp_path, capsys
):
    """(c) ATOMICITY: a failed render must never truncate the host's unit.

    Degrading to "stale but working" is recoverable — the pre-install parity
    gate reports it on the next run. Degrading to "no unit at all", or to a
    half-written one, is not.
    """
    installed = _installed_with_nine_roots(_OLD_ROOT)
    argv, output = _cli(tmp_path, output_text=installed)
    (tmp_path / "dashboard.service.template").unlink()

    assert _main(argv) != 0
    assert output.read_text(encoding="utf-8") == installed
    out = capsys.readouterr()
    assert _TAG in (out.out + out.err)


def test_main_leaves_output_byte_unchanged_when_preservation_cannot_apply(
    tmp_path, capsys
):
    """(c, second shape) A template that no longer declares a preserved name.

    The value would be read off the installed unit and dropped on the floor.
    That must be a LOUD non-zero exit with the file untouched, never a silent
    success — it is the original defect, one layer in.
    """
    installed = _installed_with_nine_roots(_OLD_ROOT)
    shrunk = "\n".join(
        line
        for line in TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith(f"Environment={_KNOWN_ROOTS}=")
    )
    argv, output = _cli(tmp_path, template_text=shrunk, output_text=installed)

    assert _main(argv) != 0
    assert output.read_text(encoding="utf-8") == installed
    out = capsys.readouterr()
    assert _KNOWN_ROOTS in (out.out + out.err)


def test_main_every_emitted_line_carries_the_log_tag(tmp_path, capsys):
    """(e) On stdout AND stderr, success AND failure.

    Load-bearing the same way test_main_every_emitted_line_carries_the_log_tag
    is for the parity checker: a shell caller routes an operator to a report BY
    TAG, and setup-host.sh's own gates treat the tag's ABSENCE as conclusive
    proof a tool never ran. An untagged line makes that reasoning unsound.
    """
    argv, _output = _cli(tmp_path, output_text=_installed_with_nine_roots(_OLD_ROOT))
    _main(argv)
    ok = capsys.readouterr()

    (tmp_path / "dashboard.service.template").unlink()
    _main(argv)
    bad = capsys.readouterr()

    emitted = [
        line
        for chunk in (ok.out, ok.err, bad.out, bad.err)
        for line in chunk.splitlines()
        if line.strip()
    ]
    assert emitted, "main() emitted nothing at all, so the tag claim is vacuous"
    for line in emitted:
        assert line.startswith(_TAG), f"untagged line: {line!r}"


def test_main_reports_what_was_preserved(tmp_path, capsys):
    """(f) The install leaves a RECORD naming the variable and its value.

    This is the property the post-install parity check structurally cannot
    supply: DASHBOARD_KNOWN_PROJECT_ROOTS is on DIVERGENCE_ALLOWLIST, so that
    gate is incapable of saying anything about its value. A tagged line naming
    what was preserved is the ONLY evidence the variable was handled at all.
    """
    argv, _output = _cli(tmp_path, output_text=_installed_with_nine_roots(_OLD_ROOT))

    _main(argv)

    report = capsys.readouterr().out
    assert _KNOWN_ROOTS in report, report
    assert NINE_ROOTS in report, report


def test_main_reports_a_fallback_to_the_rendered_default(tmp_path, capsys):
    """(f, second shape) A greenfield fallback is REPORTED, not taken silently.

    Both outcomes are legitimate; only one of them is legitimate to take without
    saying so, and nothing downstream can tell the operator which happened.
    """
    argv, _output = _cli(tmp_path)

    _main(argv)

    report = capsys.readouterr().out
    assert _KNOWN_ROOTS in report, report
    assert "absent" in report.lower(), report
