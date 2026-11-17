"""The dashboard browser's recurring HTTP pollers, and readers over its redux sources.

CLIENT_POLLERS registers every recurring HTTP poller the shipped client runs.
The readers below derive each poller's connection-reuse interval, and classify
every setInterval call site, from the live redux sources. The consumer is
tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::test_keep_alive_timeout_is_pinned_above_poll_interval.

This module holds NO test functions. Its guards live in
tests/scripts/test_dashboard_client_pollers.py. It is importable from
tests/scripts/test_*.py only because tests/scripts/conftest.py puts this
directory on sys.path.
"""

import functools
import pathlib
import re
import typing

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The browser's poll interval is DERIVED from the client source, never restated
# here.  plans/dashboard-availability-prd.md records that the dashboard polls
# every 3 seconds and that the resulting keep-alive connections "never idle out
# under a 3s poll" — which is exactly why the drain never completed before the
# graceful-shutdown bound was added.  But the invariant
# tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::
# test_keep_alive_timeout_is_pinned_above_poll_interval
# protects is RELATIONAL (keep-alive must exceed the interval at which the
# browser reuses its connection), so a constant copied out of the JS would rot
# silently: bump the poller to 6000ms and a hardcoded `3` keeps the test green
# (5 > 3) while the real invariant (5 > 6) is violated.  Parsing the live value
# instead makes that bump fail loudly, the same way
# tests/scripts/test_dashboard_service_template.py::
# test_template_renders_to_hardcoded_file renders the template rather than
# restating its contents.
#
# The browser runs more than one recurring HTTP poller, and each holds its own
# keep-alive connection and reuses it at its own interval, so the bound must
# clear the SLOWEST of them.  CLIENT_POLLERS below names the covered set,
# slowest_client_poll() reduces over it, and
# tests/scripts/test_dashboard_client_pollers.py::
# test_every_client_setinterval_is_registered_or_allowlisted makes an
# unregistered new poller a loud failure rather than a silent one.
REDUX_DIR = REPO_ROOT / "dashboard" / "src" / "dashboard" / "static" / "redux"
POLL_JS = REDUX_DIR / "data.js"

# The SECOND recurring HTTP poller: HostLoadCard's /api/load poll.  It is fully
# independent of data.js's refresh — its own setInterval, its own connection,
# its own reuse interval — so the keep-alive bound has to clear it too.
LOAD_POLL_JSX = REDUX_DIR / "tab_overview.jsx"

# The poller names its period explicitly, so anchor on that name rather than on
# whichever call site happens to consume it.  A call-site literal is gone the
# moment a poller passes a named constant instead, so bind to the declaration,
# not to the literal: it is both stabler and more precise.
#
# The name is a PARAMETER rather than a module constant because more than one
# client poller declares a period (see CLIENT_POLLERS below).  The patterns are
# otherwise byte-identical per name; in particular the ``^\s*(?:const|let|var)
# \s+`` anchor is what makes LOAD_POLL_INTERVAL_MS and POLL_INTERVAL_MS
# mutually non-matching, so one poller can never answer a lookup for another.
@functools.cache
def _int_decl_pattern(const_name: str) -> re.Pattern[str]:
    """Return the compiled ``const <const_name> = <int literal>;`` pattern."""
    return re.compile(
        rf"^\s*(?:const|let|var)\s+{re.escape(const_name)}\s*=\s*(\d+)\s*;",
        re.MULTILINE,
    )


@functools.cache
def _poll_interval_patterns(const_name: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Return the (declaration, use) regex pair for *const_name*, compiled once."""
    return (
        _int_decl_pattern(const_name),
        re.compile(rf"setInterval\([^;]*?,\s*{re.escape(const_name)}\s*\)"),
    )


def _parse_declared_int(source: str, origin: str, const_name: str) -> int:
    """Return the single integer literal *const_name* is declared with in *source*.

    The declaration half of parse_poll_interval_ms, factored out because not
    every timing constant the keep-alive bound depends on is a ``setInterval``
    period — data.js's ``JITTER_MAX_MS`` is awaited inside the request path, not
    passed to a timer, so it has no call site to bind to.  Exactly one
    declaration is still required, for the same reason: an ambiguous value is
    not a value.
    """
    decls = _int_decl_pattern(const_name).findall(source)
    assert len(decls) == 1, (
        f"Expected exactly one {const_name} = <literal> declaration in "
        f"{origin}, found {len(decls)}: {decls}. The keep-alive bound in "
        "tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::"
        "test_keep_alive_timeout_is_pinned_above_poll_interval is derived "
        "from that constant; if it was renamed, moved, or "
        "made non-literal, update the reader to identify the value explicitly "
        "rather than dropping the check."
    )
    return int(decls[0])


def parse_poll_interval_ms(
    source: str, origin: str, const_name: str = "POLL_INTERVAL_MS"
) -> int:
    """Return the period named *const_name* in *source*, in milliseconds.

    Reads the ``<const_name>`` declaration, and separately asserts that the
    constant is what actually drives a ``setInterval``.  Both halves must hold:
    a declaration nobody schedules on would let this reader measure a dead
    constant, and a scheduler whose period we cannot read would let the
    keep-alive bound be checked against nothing.  Exactly one declaration is
    required so the period can never be ambiguous.

    *const_name* defaults to data.js's ``POLL_INTERVAL_MS``, but is a parameter
    because the browser runs more than one recurring HTTP poller and each names
    its own period — tab_overview.jsx's host-load card declares
    ``LOAD_POLL_INTERVAL_MS``.  Parameterizing the existing parser rather than
    writing a second one carries both halves of the contract above to every
    poller for free.  Every message names the REQUESTED constant, so a miss on
    one poller cannot report itself as a problem with another.
    """
    _, use_re = _poll_interval_patterns(const_name)
    period = _parse_declared_int(source, origin, const_name)
    assert use_re.search(source), (
        f"{const_name} is declared in {origin} but never passed as a "
        "setInterval(...) period, so it may no longer be that poller's actual "
        "interval. Update parse_poll_interval_ms to read "
        "whatever now schedules the poll."
    )
    return period


class ClientPoller(typing.NamedTuple):
    """One recurring HTTP poller the browser runs, for keep-alive purposes.

    *what* is prose naming the poller, so a failure can say WHICH one set the
    keep-alive floor instead of just printing a number.

    *jitter_const* names a declared delay the poller adds ON TOP of its timer
    period before issuing the request, or None when it adds none.  It is part of
    the entry because the quantity the keep-alive bound needs is the worst-case
    gap between two consecutive requests on one connection, not the timer period
    — see poller_reuse_ms.
    """

    path: pathlib.Path
    const_name: str
    what: str
    jitter_const: str | None = None


# Every recurring HTTP poller in the shipped client.  Membership is the property
# that matters: an unregistered poller is one the keep-alive floor does not
# clear.  Periods are read from the live sources at run time (never restated
# here), so there is no third copy to go stale.
#
# data.js registers its JITTER_MAX_MS too, because its timer period alone
# UNDERSTATES how far apart its requests can land — refreshOne awaits
# `sleep(random() * jitterMaxMs)` inside the in-flight window (data.js:261), so
# consecutive requests on one endpoint can be POLL_INTERVAL_MS + JITTER_MAX_MS
# apart.  Deriving the floor from the bare period would be the same silent
# understatement CLIENT_POLLERS exists to remove, one level further down.
#
# A new recurring poller MUST be added here.  The completeness guard
# (tests/scripts/test_dashboard_client_pollers.py::
# test_every_client_setinterval_is_registered_or_allowlisted) fails on any
# setInterval under redux/ that is neither registered here nor in
# NON_POLLING_TIMERS, so forgetting is loud rather than silent.
CLIENT_POLLERS: tuple[ClientPoller, ...] = (
    ClientPoller(
        POLL_JS, "POLL_INTERVAL_MS", "main data refresh", jitter_const="JITTER_MAX_MS"
    ),
    ClientPoller(
        LOAD_POLL_JSX, "LOAD_POLL_INTERVAL_MS", "host-load card, /api/load"
    ),
)


def poller_reuse_ms(entry: ClientPoller) -> int:
    """Return the worst-case gap between two of *entry*'s consecutive requests.

    The keep-alive invariant is about the interval at which the browser REUSES
    a connection, which is the timer period only when the poller fires its
    request the instant the timer does.  data.js does not: refreshOne awaits a
    random jitter of up to JITTER_MAX_MS *inside* the in-flight window before
    fetching, so a zero-jitter tick followed by a max-jitter tick puts
    POLL_INTERVAL_MS + JITTER_MAX_MS between the two requests.  Summing the two
    is therefore the honest bound; using the period alone would certify a
    keep-alive that the real spacing can exceed.
    """
    source = entry.path.read_text(encoding="utf-8")
    period = parse_poll_interval_ms(
        source, str(entry.path), const_name=entry.const_name
    )
    if entry.jitter_const is None:
        return period
    jitter = _parse_declared_int(source, str(entry.path), entry.jitter_const)
    # The declaration is only meaningful if something still applies it; a jitter
    # constant left behind after the delay was removed would inflate the floor
    # off a dead number.  Two occurrences = the declaration plus at least one use.
    uses = len(re.findall(rf"\b{re.escape(entry.jitter_const)}\b", source))
    assert uses >= 2, (
        f"{entry.jitter_const} is declared in {entry.path} ({entry.what}) but "
        "never referenced again, so it may no longer delay the request. Update "
        "the CLIENT_POLLERS entry to name whatever now offsets the poll, or "
        "drop jitter_const if the poller fires on its timer."
    )
    return period + jitter


def slowest_client_poll() -> tuple[int, ClientPoller]:
    """Return (reuse_ms, poller) for the SLOWEST registered client poller.

    Returns the entry alongside the interval so callers can name which poller
    set the keep-alive floor; a bare number leaves the reader to go find it.
    The interval is the connection-REUSE interval from poller_reuse_ms, not the
    bare timer period.
    """
    measured = [(poller_reuse_ms(entry), entry) for entry in CLIENT_POLLERS]
    return max(measured, key=lambda pair: pair[0])


# Recurring client timers that are deliberately NOT pollers, keyed on
# (relpath, period-text) with the reason they are exempt.
#
# ENTRY CRITERION — the timer must open NO network connection.  If it fetches
# anything it belongs in CLIENT_POLLERS instead, because it then independently
# holds a keep-alive socket and reuses it at its own interval, which is exactly
# what the --timeout-keep-alive floor has to clear.  Parking a real poller here
# would silence the guard while re-creating the bug it exists to catch.
#
# Entries are checked for staleness too: one that matches no live call site is
# reported by tests/scripts/test_dashboard_client_pollers.py::
# test_every_client_setinterval_is_registered_or_allowlisted, so the allowlist
# cannot quietly rot into a no-op.
NON_POLLING_TIMERS: dict[tuple[str, str], str] = {
    ("redux/app.jsx", "1000"): (
        "App-wide clock tick: setInterval(() => setNow(new Date()), 1000) only "
        "advances React state so rendered times refresh. No fetch, no "
        "connection."
    ),
    ("redux/shell.jsx", "2000"): (
        "LiveFeed re-tick: setInterval(() => tick(n => n + 1), 2000) re-renders "
        "the relative `timeago` strings from the already-fetched window.DF_DATA "
        "even when it has not changed. Issues no request."
    ),
}


# Matches a `//` line comment or a `/* ... */` block comment.  Block first, so a
# `//` inside a block does not end the line-comment match early.
#
# The `(?<![:\w])` guard keeps a URL's `//` from being read as a comment start.
# The live client carries `http://www.w3.org/2000/svg` inside a template literal
# at redux/tweaks-panel.jsx:82; without the guard everything after that `//` is
# blanked, which would HIDE any call site sharing the line.
_JS_COMMENT_RE = re.compile(r"/\*.*?\*/|(?<![:\w])//[^\n]*", re.DOTALL)


def _strip_js_comments(source: str) -> str:
    """Return *source* with comments blanked out, preserving line structure.

    Comment bodies are replaced by their own newlines rather than deleted, so a
    site's line number and the file's shape survive the strip.

    Load-bearing, not cosmetic.  The live client mentions ``setInterval(...)``
    inside a ``//`` comment in tab_tasks.jsx, so a raw-text scan would report a
    call site that does not exist — training maintainers to allowlist fictions,
    and (worse) letting an allowlist entry be "satisfied" by a commented-out
    line.  This is the same hazard
    tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::_uvicorn_int_flag
    already handles from the other direction, by scoping its lookup to the
    logical ExecStart line rather than the whole file, because both unit files
    discuss the very flags it reads in prose above the command.  Scope to code,
    never to raw file text.

    Still a regex, not a JS lexer, so it stays naive about strings and regex
    literals — but the naivety is bounded where it actually bites.  The common
    real case is a URL, whose ``//`` follows a ``:``, and the pattern's
    ``(?<![:\\w])`` guard refuses to start a comment there; the live tree has
    exactly that, inside a template literal at redux/tweaks-panel.jsx:82.  A
    protocol-relative ``"//cdn…"`` or a ``//`` inside a regex literal would
    still over-strip.

    That residual is a real hole, NOT a safe direction: the stale check only
    fires for a site that ALREADY has a NON_POLLING_TIMERS entry.  A newly
    added poller sharing a line with an over-stripped ``//`` has no entry to go
    stale, so it would be hidden with nothing to report it — precisely the
    silent-uncovered-poller failure that
    tests/scripts/test_dashboard_client_pollers.py::
    test_every_client_setinterval_is_registered_or_allowlisted exists to
    prevent.  If such a line ever appears, extend the stripper to skip
    string/template spans rather than working around it.
    """
    return _JS_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), source)


def _setinterval_period_args(source: str) -> list[str]:
    """Return the period argument text of every ``setInterval(`` call in *source*.

    Scans the argument list with a balanced-paren walk and takes the text after
    the LAST top-level comma.  A naive ``setInterval\\([^)]*\\)`` regex stops at
    the first ``)``, which both live non-polling timers contain in their
    callbacks — ``setInterval(() => setNow(new Date()), 1000)`` and
    ``setInterval(() => tick(n => n + 1), 2000)`` — so it would yield a garbled
    period for exactly the sites this scanner exists to classify.
    """
    periods: list[str] = []
    for match in re.finditer(r"\bsetInterval\s*\(", source):
        depth = 1
        last_comma = None
        idx = match.end()
        while idx < len(source) and depth:
            char = source[idx]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if not depth:
                    break
            elif char == "," and depth == 1:
                last_comma = idx
            idx += 1
        if depth and idx >= len(source):
            # Unbalanced call — report the whole tail rather than silently
            # dropping a site the maintainer needs to see.
            periods.append(source[match.end():].strip())
            continue
        arg_start = match.end() if last_comma is None else last_comma + 1
        periods.append(source[arg_start:idx].strip())
    return periods


def site_relpath(path: pathlib.Path) -> str:
    """Return the short ``<dir>/<file>`` key a call site is reported under.

    Two components, so the key is stable across checkouts and worktrees (an
    absolute path would make every allowlist entry machine-specific) while still
    being unambiguous and greppable within the client tree.
    """
    return f"{path.parent.name}/{path.name}"


def unclassified_setinterval_sites(
    paths: typing.Iterable[pathlib.Path],
    known_sites: typing.AbstractSet[tuple[str, str]],
    allowlist: typing.Mapping[tuple[str, str], str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Sort every live ``setInterval`` under *paths* into known / allowed / new.

    *known_sites* is keyed on the ``(relpath, const_name)`` PAIR, not on the
    constant name alone.  Matching on the name alone would wave through a
    ``setInterval(fetchX, POLL_INTERVAL_MS)`` in some other file — covered, says
    the guard; never read, says slowest_client_poll, which only ever opens the
    paths CLIENT_POLLERS registers — re-creating the silent-coverage hole one
    level down.  That is reachable rather than theoretical for .jsx:
    they load as ``type="text/babel"`` and Babel standalone evaluates each
    separately, so a same-named top-level ``const`` in a second .jsx need not
    collide with data.js's global (dashboard/tests/js/classic_script_scope.test.mjs).

    Returns ``(unclassified, stale)``:

    * *unclassified* — ``(relpath, period_text)`` for each call site that is
      neither a *known_sites* pair (a registered poller, in the file it is
      registered for) nor an *allowlist* key.  These are the sites nothing has
      decided about, so the keep-alive floor may not clear them.
    * *stale* — *allowlist* keys matching no live call site.  An entry that
      matches nothing checks nothing, and rots the gate into a green no-op —
      the same staleness reasoning as
      test_registry_keys_are_all_declared_in_the_committed_units in
      tests/scripts/test_check_dashboard_unit_parity.py.  Reporting it also
      keeps a commented-out site from satisfying its own allowlist entry.

    Both lists are returned rather than asserted here so the caller can name the
    concrete remedy for each; see _strip_js_comments and
    _setinterval_period_args for why the scan is done over comment-stripped
    source with a balanced-paren walk.
    """
    unclassified: list[tuple[str, str]] = []
    live: set[tuple[str, str]] = set()
    for path in paths:
        relpath = site_relpath(path)
        code = _strip_js_comments(path.read_text(encoding="utf-8"))
        for period in _setinterval_period_args(code):
            site = (relpath, period)
            live.add(site)
            if site in known_sites or site in allowlist:
                continue
            unclassified.append(site)
    stale = sorted(key for key in allowlist if key not in live)
    return unclassified, stale
