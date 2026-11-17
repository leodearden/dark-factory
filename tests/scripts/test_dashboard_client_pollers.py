"""Tests for the dashboard client's recurring-poller registry and setInterval scanner.

The readers and the registry live in tests/scripts/dashboard_client_pollers.py.
Each reader is pinned here against synthetic sources, and live-tree guards
check that the shipped redux client is still fully covered.
"""

import pathlib

import pytest
from dashboard_client_pollers import (
    CLIENT_POLLERS,
    LOAD_POLL_JSX,
    NON_POLLING_TIMERS,
    POLL_JS,
    REDUX_DIR,
    ClientPoller,
    parse_poll_interval_ms,
    poller_reuse_ms,
    site_relpath,
    slowest_client_poll,
    unclassified_setinterval_sites,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def test_poll_interval_parser_requires_exactly_one_interval() -> None:
    """parse_poll_interval_ms must fail loudly on zero or ambiguous matches.

    The keep-alive bound is derived from this parse, so a silent miss would turn
    the relational assertion into a no-op against a value nobody supplied.
    """
    assert (
        parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\n"
            "const handle = setInterval(() => pollTick(opts), POLL_INTERVAL_MS);\n",
            "synthetic",
        )
        == 3000
    )

    with pytest.raises(AssertionError, match="found 0"):
        parse_poll_interval_ms("refreshDFData();\n", "synthetic")

    with pytest.raises(AssertionError, match="found 2"):
        parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\n"
            "const POLL_INTERVAL_MS = 9000;\n"
            "setInterval(tick, POLL_INTERVAL_MS);\n",
            "synthetic",
        )

    # Declared but nothing schedules on it: the constant may be vestigial, so the
    # keep-alive bound would be measured against a period the browser never uses.
    with pytest.raises(AssertionError, match="never passed as a setInterval"):
        parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\nsetInterval(tick, 250);\n", "synthetic"
        )


def test_poll_interval_parser_reads_an_arbitrary_constant_name() -> None:
    """parse_poll_interval_ms must parse ANY named period, not just POLL_INTERVAL_MS.

    More than one client poller declares a period — data.js's POLL_INTERVAL_MS
    (the main data refresh) and tab_overview.jsx's LOAD_POLL_INTERVAL_MS (the
    host-load card's /api/load poll) — and the keep-alive lower bound is derived
    from the SLOWEST of them.  Rather than growing a second bespoke parser, the
    existing declaration-anchored one takes the constant name as a parameter, so
    both halves of its contract (exactly-one-declaration, and the declared
    constant actually schedules a setInterval) carry over to every poller for
    free instead of being re-derived per poller.

    Two properties beyond "it parses" are pinned here.  The failure messages must
    name the REQUESTED constant: a LOAD_POLL_INTERVAL_MS miss that reported
    itself as a POLL_INTERVAL_MS problem would send the fixer to the wrong file.
    And the two names must not cross-match, or a max over the poller registry
    could be computed twice from the same source with the other poller silently
    dropping out — the exact silent-coverage failure this registry exists to
    remove.
    """
    assert (
        parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\n"
            "const id = setInterval(fetchLoad, LOAD_POLL_INTERVAL_MS);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )
        == 5000
    )

    # Both halves of the contract still hold for the requested name, and each
    # failure names the constant it was actually looking for.
    with pytest.raises(AssertionError, match="found 0") as absent:
        parse_poll_interval_ms(
            "fetchLoad();\n", "synthetic", const_name="LOAD_POLL_INTERVAL_MS"
        )
    assert "LOAD_POLL_INTERVAL_MS" in str(absent.value)

    with pytest.raises(AssertionError, match="found 2") as ambiguous:
        parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\n"
            "const LOAD_POLL_INTERVAL_MS = 9000;\n"
            "setInterval(fetchLoad, LOAD_POLL_INTERVAL_MS);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )
    assert "LOAD_POLL_INTERVAL_MS" in str(ambiguous.value)

    with pytest.raises(AssertionError, match="never passed as a setInterval") as dead:
        parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\nsetInterval(fetchLoad, 250);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )
    assert "LOAD_POLL_INTERVAL_MS" in str(dead.value)

    # The two live names must not cross-match in EITHER direction. The
    # ``^\s*(?:const|let|var)\s+`` anchor is what keeps LOAD_POLL_INTERVAL_MS
    # from satisfying a POLL_INTERVAL_MS lookup by suffix; a substring match
    # would let one poller answer for both.
    with pytest.raises(AssertionError, match="found 0"):
        parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\n"
            "setInterval(() => pollTick(opts), POLL_INTERVAL_MS);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )

    with pytest.raises(AssertionError, match="found 0"):
        parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\n"
            "setInterval(fetchLoad, LOAD_POLL_INTERVAL_MS);\n",
            "synthetic",
        )


def _write_synthetic_client(path: pathlib.Path, source: str) -> pathlib.Path:
    """Write a synthetic client-side source file for scanner-verification tests.

    The scanner's own behaviour is pinned against sources this module
    controls, so only the live-tree guards are coupled to the real dashboard
    client.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_setinterval_scanner_classifies_every_call_site(
    tmp_path: pathlib.Path,
) -> None:
    """unclassified_setinterval_sites must sort live timers into known/allowed/new.

    This is the guard-the-guard test for the completeness check: a scanner that
    silently reported nothing would turn
    test_every_client_setinterval_is_registered_or_allowlisted into a green
    no-op, which is the exact failure mode that let a second poller go
    unnoticed in the first place.

    Two hazards get their own cases.  Comment immunity: the live tree mentions
    ``setInterval(...)`` inside a ``//`` comment at tab_tasks.jsx, so a raw-text
    scan would report a phantom site and train maintainers to allowlist
    fictions.  That is the same trap
    tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::_uvicorn_int_flag
    already handles from the other direction by scoping to the logical
    ExecStart line rather than the whole file (see
    tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::
    test_uvicorn_flag_lookup_is_scoped_to_exec_start).  And
    nested calls: a naive ``setInterval\\([^)]*\\)`` match stops at the FIRST
    ``)``, so ``setInterval(() => tick(n => n + 1), 2000)`` would yield a
    garbled period — both live non-polling timers have exactly this shape.

    Two more cases pin the edges of "classified".  A registered constant used
    in an UNregistered file must still be reported: coverage is the
    (file, constant) pair, because slowest_client_poll only ever reads the
    paths CLIENT_POLLERS names, so a name-only match would claim coverage
    nothing provides.  And a URL's ``//`` must not start a comment, or a real
    call site sharing that line is blanked out of the scan entirely — with no
    allowlist entry to go stale and report it.
    """
    redux = tmp_path / "redux"

    # Registered: driven by a known poller constant, in its registered file.
    registered = _write_synthetic_client(
        redux / "data.js",
        "const POLL_INTERVAL_MS = 3000;\n"
        "setInterval(() => pollTick(opts), POLL_INTERVAL_MS);\n",
    )
    # Same constant NAME, different (unregistered) file — must NOT be waved
    # through: nothing reads this file's declaration, so nothing bounds it.
    impostor = _write_synthetic_client(
        redux / "tab_impostor.jsx",
        "const POLL_INTERVAL_MS = 9000;\n"
        "setInterval(fetchOther, POLL_INTERVAL_MS);\n",
    )
    # A live call site sharing a line with a URL inside a string literal.
    urly = _write_synthetic_client(
        redux / "tab_urly.jsx",
        "const u = 'https://x/y'; setInterval(fetchUrly, 8000);\n",
    )
    # Allowlisted: a pure UI tick, and a NESTED call in the callback.
    allowlisted = _write_synthetic_client(
        redux / "app.jsx",
        "setInterval(() => setNow(new Date()), 1000);\n",
    )
    # Unregistered and unallowlisted — must be reported.
    unknown = _write_synthetic_client(
        redux / "tab_thing.jsx",
        "setInterval(fetchThing, 4000);\n",
    )
    # Also unclassified, and nested two deep: pins the balanced-paren scan.
    nested = _write_synthetic_client(
        redux / "shell.jsx",
        "setInterval(() => tick(n => n + 1), 2000);\n",
    )
    # Comments only: a `//` mention and a `/* */` block. Neither is live code.
    commented = _write_synthetic_client(
        redux / "tab_tasks.jsx",
        "// ...`setInterval(() => setNow(...), 1000)`, unrelated\n"
        "/* legacy, removed:\n"
        "setInterval(fetchOld, 7000);\n"
        "*/\n",
    )

    allowlist = {
        ("redux/app.jsx", "1000"): "clock tick — opens no connection",
        # Points at a call site that now exists only inside a comment.
        ("redux/tab_tasks.jsx", "1000"): "checks nothing anymore",
    }
    unclassified, stale = unclassified_setinterval_sites(
        [registered, impostor, urly, allowlisted, unknown, nested, commented],
        known_sites={("redux/data.js", "POLL_INTERVAL_MS")},
        allowlist=allowlist,
    )

    # (a) known (file, constant) pair, (b) allowlisted, (d) comment-only: none
    # reported.  (c) every genuinely new timer is, named with file and period —
    # including the name-collision impostor and the site sharing a URL's line.
    assert sorted(unclassified) == [
        ("redux/shell.jsx", "2000"),
        ("redux/tab_impostor.jsx", "POLL_INTERVAL_MS"),
        ("redux/tab_thing.jsx", "4000"),
        ("redux/tab_urly.jsx", "8000"),
    ], (
        "Scanner must report exactly the unregistered, unallowlisted call sites "
        f"— with their periods — and nothing else; got {sorted(unclassified)}. "
        "A phantom from a commented-out site would train maintainers to "
        "allowlist fictions; a missed real site is a poller the keep-alive "
        "floor does not clear."
    )

    # (e) an allowlist entry whose site is now only a comment is STALE, not
    # silently satisfied — otherwise the allowlist rots into a no-op gate.
    assert stale == [("redux/tab_tasks.jsx", "1000")], (
        "An allowlist entry matching no live call site must be reported stale; "
        f"got {stale}."
    )


def test_client_poll_interval_is_readable_from_the_shipped_client() -> None:
    """The live client source must still yield a poll interval to compare against.

    Guards the derivation itself: if data.js moves or its poller is rewritten,
    this fails here with a clear cause rather than silently weakening
    tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::
    test_keep_alive_timeout_is_pinned_above_poll_interval.
    """
    assert POLL_JS.is_file(), (
        f"Client poll source not found at {POLL_JS}. The keep-alive lower bound "
        "is derived from it; update POLL_JS if the client moved."
    )
    assert parse_poll_interval_ms(POLL_JS.read_text(encoding="utf-8"), str(POLL_JS)) > 0


def test_host_load_poller_declares_a_named_interval() -> None:
    """The host-load card's /api/load poller must expose a parseable period too.

    HostLoadCard in tab_overview.jsx runs an INDEPENDENT second HTTP poller: it
    fetches /api/load on its own setInterval, entirely separate from data.js's
    data refresh.  It therefore holds its own keep-alive connection and reuses
    it at its own interval, so its period is a load-bearing input to the
    keep-alive lower bound and must be derivable — a poller whose period cannot
    be read is a poller the bound cannot clear.

    Naming the period rather than leaving a bare ``setInterval(fetchLoad, 5000)``
    literal is what makes that derivation stable, for the reason in the comment
    above tests/scripts/dashboard_client_pollers.py::_int_decl_pattern.  The
    declaration also gives the JSX a place to carry the "this feeds the systemd
    keep-alive bound" warning.
    """
    assert LOAD_POLL_JSX.is_file(), (
        f"Host-load poll source not found at {LOAD_POLL_JSX}. The keep-alive "
        "lower bound is derived from it; update LOAD_POLL_JSX if the host-load "
        "card moved."
    )
    assert (
        parse_poll_interval_ms(
            LOAD_POLL_JSX.read_text(encoding="utf-8"),
            str(LOAD_POLL_JSX),
            const_name="LOAD_POLL_INTERVAL_MS",
        )
        > 0
    )


def test_poller_reuse_interval_adds_declared_jitter(tmp_path: pathlib.Path) -> None:
    """A poller's reuse interval must include any delay it adds before fetching.

    Pinned against synthetic sources so the arithmetic is checked independently
    of whatever the live client currently declares.  The keep-alive floor is
    about how long a connection sits idle between two requests, and a poller
    that sleeps up to J ms inside its request path spaces consecutive requests
    by up to period + J — so deriving the floor from the timer period alone
    understates it, silently, in the direction that certifies a too-low
    keep-alive.  data.js does exactly this (JITTER_MAX_MS, awaited inside the
    in-flight window at data.js:261).
    """
    plain = _write_synthetic_client(
        tmp_path / "plain.js",
        "const POLL_INTERVAL_MS = 3000;\nsetInterval(tick, POLL_INTERVAL_MS);\n",
    )
    assert (
        poller_reuse_ms(ClientPoller(plain, "POLL_INTERVAL_MS", "no jitter")) == 3000
    ), "A poller that fires on its timer reuses at exactly its period."

    jittered = _write_synthetic_client(
        tmp_path / "jittered.js",
        "const JITTER_MAX_MS = 1500;\n"
        "const POLL_INTERVAL_MS = 3000;\n"
        "setInterval(tick, POLL_INTERVAL_MS);\n"
        "async function go() { await sleep(random() * JITTER_MAX_MS); }\n",
    )
    assert (
        poller_reuse_ms(
            ClientPoller(
                jittered, "POLL_INTERVAL_MS", "jittered", jitter_const="JITTER_MAX_MS"
            )
        )
        == 4500
    ), (
        "Worst-case spacing is period + jitter: a zero-jitter tick followed by a "
        "max-jitter tick puts POLL_INTERVAL_MS + JITTER_MAX_MS between requests."
    )

    # A jitter constant nothing applies would inflate the floor off a dead
    # number — as wrong as omitting it, just in the other direction.
    dead = _write_synthetic_client(
        tmp_path / "dead.js",
        "const JITTER_MAX_MS = 1500;\n"
        "const POLL_INTERVAL_MS = 3000;\n"
        "setInterval(tick, POLL_INTERVAL_MS);\n",
    )
    with pytest.raises(AssertionError, match="never referenced again") as unused:
        poller_reuse_ms(
            ClientPoller(
                dead, "POLL_INTERVAL_MS", "dead jitter", jitter_const="JITTER_MAX_MS"
            )
        )
    assert "JITTER_MAX_MS" in str(unused.value)


def test_slowest_client_poll_is_derived_from_every_registered_poller() -> None:
    """The keep-alive floor must come from the SLOWEST poller, not from data.js.

    CLIENT_POLLERS is the explicit statement of which recurring HTTP pollers the
    keep-alive bound covers, and slowest_client_poll() reduces it with max()
    — because each poller independently reuses its own connection at its own
    interval, so a bound that clears only the fastest one still closes the
    slower socket in the gap between its polls.

    The registry gets the same staleness guard as
    test_registry_keys_are_all_declared_in_the_committed_units in
    tests/scripts/test_check_dashboard_unit_parity.py: every registered path must
    exist and every registered constant must still parse, so a moved file or a
    renamed constant fails loudly here instead of silently dropping a poller out
    of the max and quietly lowering the floor.

    What is pinned is MEMBERSHIP: at least two entries, both specific pollers
    present, and the reduction returning one of them.  The relative ORDER of
    the two periods is deliberately left unpinned: an order assertion would
    also fire on a perfectly legitimate retune (dropping /api/load to 2s, or
    raising the data refresh past 5s) that leaves the registry complete and the
    bound correct.  A guard that reports a false cause is worse than one that
    reports nothing.
    """
    assert len(CLIENT_POLLERS) >= 2, (
        f"CLIENT_POLLERS has {len(CLIENT_POLLERS)} entr(y/ies); the shipped "
        "client runs at least two independent recurring HTTP pollers, so a "
        "shorter registry means one has been dropped and the keep-alive floor "
        "no longer clears it."
    )
    registered = {(entry.path, entry.const_name) for entry in CLIENT_POLLERS}
    assert (POLL_JS, "POLL_INTERVAL_MS") in registered, (
        "data.js's main data refresh is missing from CLIENT_POLLERS; the "
        "keep-alive floor would stop covering it."
    )
    assert (LOAD_POLL_JSX, "LOAD_POLL_INTERVAL_MS") in registered, (
        "tab_overview.jsx's /api/load host-load poll is missing from "
        "CLIENT_POLLERS. It is an independent recurring HTTP poller and holds "
        "its own keep-alive connection, so the floor must clear it too."
    )

    reuses = []
    for entry in CLIENT_POLLERS:
        assert entry.path.is_file(), (
            f"CLIENT_POLLERS registers {entry.path} ({entry.what}) but that file "
            "does not exist. Update the registry entry rather than leaving a "
            "poller silently uncovered by the keep-alive floor."
        )
        period = parse_poll_interval_ms(
            entry.path.read_text(encoding="utf-8"),
            str(entry.path),
            const_name=entry.const_name,
        )
        assert period > 0
        reuse = poller_reuse_ms(entry)
        assert reuse >= period, (
            f"{entry.what}'s connection-reuse interval ({reuse}ms) is below its "
            f"own timer period ({period}ms), which is incoherent — jitter can "
            "only push consecutive requests further apart, never closer."
        )
        reuses.append(reuse)

    # data.js's jitter must stay wired into the registry, not just exist in the
    # source: dropping jitter_const would put the derivation back to the bare
    # period and re-open the understatement (see poller_reuse_ms).
    data_entry = next(e for e in CLIENT_POLLERS if e.path == POLL_JS)
    assert data_entry.jitter_const, (
        "data.js's CLIENT_POLLERS entry no longer names a jitter constant. Its "
        "refreshOne awaits a random delay inside the in-flight window, so its "
        "requests land further apart than POLL_INTERVAL_MS; without "
        "jitter_const the keep-alive floor is derived from a gap smaller than "
        "the real one. Re-point jitter_const at whatever now delays the fetch, "
        "or remove it only once the delay itself is gone."
    )
    assert poller_reuse_ms(data_entry) > parse_poll_interval_ms(
        POLL_JS.read_text(encoding="utf-8"), str(POLL_JS)
    ), (
        "data.js's reuse interval equals its bare poll period, so the declared "
        f"jitter ({data_entry.jitter_const}) is contributing nothing to the "
        "keep-alive floor."
    )

    slowest_ms, slowest = slowest_client_poll()
    assert slowest_ms == max(reuses), (
        "slowest_client_poll() must be the max over every registered poller's "
        f"connection-reuse interval; got {slowest_ms} against {reuses}."
    )
    assert slowest in CLIENT_POLLERS, (
        f"slowest_client_poll() returned {slowest}, which is not a registered "
        "entry — the floor must be attributable to a poller the registry names."
    )


def test_every_client_setinterval_is_registered_or_allowlisted() -> None:
    """Every recurring timer in the client must be classified, not merely absent.

    An unregistered poller is one the keep-alive floor does not clear, and
    nothing else would notice it.

    So membership is checked from the CLIENT side, not just the registry side:
    every setInterval under the shipped redux tree must be accounted for by one
    of two explicit decisions, and an unclassified one fails loudly with both
    remedies named.  Files are globbed live rather than enumerated, so a new
    client file is covered the moment it lands rather than when someone
    remembers to add it here.

    ``.html`` is in the sweep alongside ``.js``/``.jsx`` because index.html sits
    in this same directory and is where every ``<script>`` tag lives: an inline
    ``<script>setInterval(...)</script>`` there would otherwise be exactly as
    invisible as the /api/load poller was.  The comment-stripper and
    balanced-paren scanner read inline script text unchanged.

    Coverage is asserted per (file, constant) pair rather than per constant
    name — see unclassified_setinterval_sites for why a name-only match would
    claim coverage that slowest_client_poll does not actually provide.
    """
    sources = sorted(
        path
        for path in REDUX_DIR.rglob("*")
        if path.suffix in (".js", ".jsx", ".html") and path.is_file()
    )
    assert sources, (
        f"No .js/.jsx/.html sources found under {REDUX_DIR}. This guard would "
        "pass vacuously — update REDUX_DIR to wherever the client moved rather "
        "than leaving every client timer unchecked."
    )

    unclassified, stale = unclassified_setinterval_sites(
        sources,
        known_sites={
            (site_relpath(entry.path), entry.const_name) for entry in CLIENT_POLLERS
        },
        allowlist=NON_POLLING_TIMERS,
    )

    assert not unclassified, (
        "Unclassified setInterval call site(s) in the dashboard client: "
        f"{unclassified}. Every recurring timer must be an explicit decision, "
        "because a timer that performs HTTP holds its own keep-alive connection "
        "and reuses it at its own interval — which is how tab_overview.jsx's 5s "
        "/api/load poll went unnoticed while the keep-alive floor was derived "
        "from data.js alone.\n"
        "  - If it PERFORMS HTTP: name its period as a *_POLL_INTERVAL_MS "
        "constant and register it in CLIENT_POLLERS. If it is the slowest "
        "poller this raises the --timeout-keep-alive floor, and "
        "tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::"
        "test_keep_alive_timeout_is_pinned_above_poll_interval will say so.\n"
        "  - If it is a pure UI tick that opens NO connection: add it to "
        "NON_POLLING_TIMERS with a reason."
    )
    assert not stale, (
        f"NON_POLLING_TIMERS entries matching no live call site: {stale}. "
        "An entry that matches nothing checks nothing, so it silently rots this "
        "gate — remove it, or fix it to match the timer it was meant to cover. "
        "Note a commented-out call site does not count as live."
    )
