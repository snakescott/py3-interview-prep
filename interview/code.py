import more_itertools
import pytest


# AOC 2024 day 2
def is_safe(l: list[int]) -> bool:
    if len(l) <= 1:
        return True

    # Can do this more space efficiently at the cost of code complexity
    deltas = [w[0] - w[1] for w in more_itertools.sliding_window(l, 2)]
    smallest_delta = min(deltas)
    largest_delta = max(deltas)

    # Handles cases
    # * with no difference (smallest or largest delta is zero)
    # * smallest/largest different sign (not strictly inc/dec)
    # * abs difference > 3
    if smallest_delta * largest_delta <= 0 or abs(smallest_delta) > 3 or abs(largest_delta) > 3:
        return False

    return True


@pytest.mark.parametrize(
    ("input", "expected"),
    (  # format
        ([7, 6, 4, 2, 1], True),
        ([], True),
        ([1], True),
        ([1, 2, 7, 8, 9], False),
        ([1, 3, 2, 4, 5], False),
    ),
)
def test_is_safe(input: list[int], expected: bool) -> None:
    assert is_safe(input) == expected


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
def test_hello_world(input: int, expected: int) -> None:
    assert times_two(input) == expected


# Allows invocation as a main file, e.g. if using inside coderpad or similar
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
