import pytest

@pytest.mark.integration
def test_marked_integration():
    assert True

def test_plain():
    assert True
