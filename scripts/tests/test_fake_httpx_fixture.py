"""Contract tests for the shared ``install_fake_httpx`` fixture (task 3376).

These assert real runtime behaviour — ``sys.modules`` mutation and its
restoration at teardown — not documentation.  The fixture deduplicates six
copies of the same fake-httpx idiom across four files in this directory; if it
silently stopped shadowing the real module, or silently leaked the stub past
teardown, every one of those call-sites would degrade quietly.  Hence a direct
test of the fixture itself rather than trusting its six consumers.

Ordering matters: ``test_install_fake_httpx_is_restored_after_teardown`` must
run AFTER a test that used the fixture, so it observes the post-teardown state.
pytest collects in file order, so it is placed last.
"""
import sys

# Imported eagerly, at module scope, for two reasons.  (1) It is the executable
# proof of this file's premise: httpx IS importable here — a direct dependency
# of `shared` (shared/pyproject.toml, `httpx>=0.27`, task 2965) — so if that
# dependency were ever dropped, collection of this file fails loudly instead of
# the suite quietly agreeing with a stale "httpx is not installed" claim.
# (2) It puts the REAL module in sys.modules before any fixture runs, so the
# teardown-restoration test below compares against a known object rather than
# taking a vacuous "key was absent" branch.
import httpx as _real_httpx


def test_install_fake_httpx_shadows_the_real_module_for_lazy_imports(
    install_fake_httpx,
):
    """The stub must satisfy a function-local ``import httpx``, not just a dict read.

    All six production call-sites reach httpx through a lazy, function-local
    ``import httpx`` (``default_status_fetcher``, ``_default_poster``, ...).
    Asserting only ``sys.modules['httpx'] is fake`` would also pass against a
    stub that the real import machinery bypasses, so the identity check is made
    on the binding produced by an actual ``import`` statement.
    """
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return 'sentinel-response'

    fake = install_fake_httpx(post=_spy)

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


def test_install_fake_httpx_accepts_post_positionally(install_fake_httpx):
    """The six call-sites pass the poster positionally; pin that spelling."""

    def _spy(*args, **kwargs):
        return None

    fake = install_fake_httpx(_spy)

    assert fake.post is _spy
    assert sys.modules['httpx'] is fake


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
    assert sys.modules.get('httpx') is _real_httpx, (
        'a stub module leaked past fixture teardown'
    )

    # And a fresh lazy import — the shape the production call-sites use — must
    # reach the real library again, not a husk left behind by the stub.
    def _lazy_import_site():
        import httpx

        return httpx

    assert _lazy_import_site() is _real_httpx
    assert hasattr(_real_httpx, 'Client')
