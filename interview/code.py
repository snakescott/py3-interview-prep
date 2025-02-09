import pytest


# AOC 2024 day 1
def distance(a: list[int], b: list[int]) -> int:
    a = sorted(a)
    b = sorted(b)
    return sum(abs(x[0] - x[1]) for x in zip(a, b, strict=True))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    (  # format
        ([], [], 0),
        ([1], [2], 1),
        ([2], [1], 1),
        ([3, 4, 2, 1, 3, 3], [4, 3, 5, 3, 9, 3], 11),
        ([1], [], None),
    ),
)
def test_distance(a, b, expected):
    if len(a) == len(b):
        assert distance(a, b) == expected
    else:
        with pytest.raises(ValueError, match=r"argument .* is shorter"):
            distance(a, b)


def times_two(x: int) -> int:
    return x * 2


@pytest.mark.parametrize(
    ("input", "expected"),
    (  # format
        (0, 0),
        (1, 2),
        (-2, -4),
    ),
)
def test_hello_world(input, expected):
    assert times_two(input) == expected
