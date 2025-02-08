import pytest


def times_two(x: int) -> int:
    return x * 2


@pytest.mark.parametrize(("input", "expected"), ((0, 0), (1, 2), (-2, -4)))
def test_hello_world(input, expected):
    assert times_two(input) == expected
