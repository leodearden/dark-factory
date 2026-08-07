"""Contract tests for the shared ``install_fake_httpx`` fixture (task 3376).

These assert real runtime behaviour — ``sys.modules`` mutation and its
restoration at teardown — not documentation.  The fixture deduplicates six
copies of the same fake-httpx idiom across four files in this directory; if it
silently stopped shadowing the real module, or silently leaked the stub past
teardown, every one of those call-sites would degrade quietly.  Hence a direct
test of the fixture itself rather than trusting its six consumers.

Ordering matters: ``test_install_fake_httpx_is_restored_after_teardown`` must
run AFTER a test that used the fixture, so it observes the post-teardown state.
pytest collects in file order, so it is placed last — and, because collection
order is not something this file can enforce (``-k``, ``--lf``, a bare node id
or a future xdist/randomly plugin all break it), it SKIPS with a reason rather
than passing vacuously when no stub was installed ahead of it.
"""
import sys
from types import ModuleType

# Imported eagerly, at module scope, for two reasons.  (1) It is the executable
# proof of this file's premise: httpx IS importable here — a direct dependency
# of `shared` (shared/pyproject.toml, `httpx>=0.27`, task 2965) — so if that
# dependency were ever dropped, collection of this file fails loudly instead of
# the suite quietly agreeing with a stale "httpx is not installed" claim.
# (2) It puts the REAL module in sys.modules before any fixture runs, so the
# teardown-restoration test below compares against a known object rather than
# taking a vacuous "key was absent" branch.
import httpx as _real_httpx
import pytest

# Appended to by every test here that installs a stub; read by the
# teardown-restoration test, which can only observe a restoration that actually
# happened.  A list (rather than a `global` flag) keeps the mutation local to
# the appending test.
_INSTALLED_STUBS: list[ModuleType] = []


def test_install_fake_httpx_shadows_the_real_module_for_lazy_imports(
    install_fake_httpx,
):
    """The stub must satisfy a function-local ``import httpx``, not just a dict read.

    The four production call-sites this fixture serves each reach httpx through
    a lazy, function-local ``import httpx``: ``census._post_mcp_tool_call``,
    ``census_trigger.default_status_fetcher``, ``nightly._default_poster`` and
    ``check_transcript_persistence._default_poster``.  ("Six" elsewhere in this
    file counts the deduplicated *test* copies, not production functions.)
    Asserting only ``sys.modules['httpx'] is fake`` would also pass against a
    stub that the real import machinery bypasses, so the identity check is made
    on the binding produced by an actual ``import`` statement.
    """
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return 'sentinel-response'

    fake = install_fake_httpx(post=_spy)
    _INSTALLED_STUBS.append(fake)

    assert fake.post is _spy
    assert fake.__name__ == 'httpx'
    assert sys.modules['httpx'] is fake

    def _lazy_import_site():
        import httpx

        return httpx

    resolved = _lazy_import_site()
    assert resolved is fake, 'a function-local `import httpx` must resolve to the stub'

    assert resolved.post('http://example.invalid', json={'k': 'v'}) == 'sentinel-response'
    assert calls == [(('http://example.invalid',), {'json': {'k': 'v'}})]

    # The call-sites pass the poster positionally; pin that spelling here rather
    # than in a separate test that could only fail alongside the assertions above.
    positional = install_fake_httpx(_spy)
    _INSTALLED_STUBS.append(positional)
    assert positional.post is _spy
    assert sys.modules['httpx'] is positional


def test_install_fake_httpx_fails_loudly_on_an_unstubbed_attribute(install_fake_httpx):
    """A miss must escape ``except Exception``, not be swallowed as a fake failure.

    The stub exposes only ``post``.  That suffices for all four consumers today,
    but ``default_status_fetcher`` wraps *any* ``Exception`` into
    ``StatusFetchUnavailable``, so a production change adding e.g.
    ``except httpx.HTTPError:`` would otherwise keep
    ``test_default_status_fetcher_raises_status_fetch_unavailable_when_unreachable``
    green off a plain ``AttributeError`` — passing for the wrong reason.  The
    fixture therefore reports a miss as a ``BaseException`` (``pytest.fail``),
    which such a handler cannot eat.
    """
    fake = install_fake_httpx(lambda *args, **kwargs: None)
    _INSTALLED_STUBS.append(fake)

    try:
        # The bare attribute ACCESS is the probe; bind it so the access is not a
        # bare (B018 "useless") expression statement. Nothing reads the value —
        # every meaningful path below leaves via one of the handlers.
        _ = fake.HTTPError
    except Exception as exc:
        raise AssertionError(
            'an unstubbed attribute raised a swallowable Exception '
            f'({exc!r}); production `except Exception` handlers would eat it'
        ) from exc
    except BaseException as exc:  # noqa: B036 - the whole point: not an Exception
        assert 'HTTPError' in str(exc)
        assert 'install_fake_httpx' in str(exc), 'the miss must name the fixture to extend'
    else:
        raise AssertionError('an unstubbed httpx attribute resolved silently')

    # Dunders stay ordinary AttributeErrors so import machinery, inspect and
    # pytest introspection can probe the stub without tripping the loud path.
    assert getattr(fake, '__file__', 'absent') == 'absent'


def test_install_fake_httpx_is_restored_after_teardown():
    """After the fixture-using tests above, ``httpx`` must be the GENUINE package.

    This is what forces ``monkeypatch.setitem`` over a raw
    ``sys.modules['httpx'] = ...`` assignment: a raw assignment leaves the stub
    installed for the rest of the session, poisoning any later test that
    legitimately imports httpx.

    The module-level ``import httpx as _real_httpx`` guarantees the key was
    populated with the real module before any fixture ran, so this is an exact
    identity check and cannot pass by way of a "key was absent" shortcut.
    """
    if not _INSTALLED_STUBS:
        pytest.skip(
            'inconclusive in isolation: this test observes the teardown of a stub '
            'installed by an earlier test in this file — run the whole file'
        )

    assert sys.modules.get('httpx') is _real_httpx, (
        'a stub module leaked past fixture teardown'
    )
    assert sys.modules.get('httpx') not in _INSTALLED_STUBS

    # And a fresh lazy import — the shape the production call-sites use — must
    # reach the real library again, not a husk left behind by the stub.
    def _lazy_import_site():
        import httpx

        return httpx

    assert _lazy_import_site() is _real_httpx
    assert hasattr(_real_httpx, 'Client')
