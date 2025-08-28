import pytest


def test_import_porrest():
    try:
        import porrest           # noqa
    except ImportError:
        pytest.fail('import porrest failed')
