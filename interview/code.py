import re
from collections import defaultdict
from collections.abc import Iterable

import more_itertools
import pytest

# AOC 2024 day 5
EXAMPLE_D4 = """47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13""".split("\n")


def is_ordered(rules: list[str], updates: str):
    not_after: dict[int, set[int]] = defaultdict(lambda: set())
    for rule in rules:
        a, b = map(int, rule.split("|"))
        not_after[a].add(b)

    seen: set[int] = set()
    for update in map(int, updates.split(",")):
        if seen.intersection(not_after[update]):
            return False
        seen.add(update)
    return True


@pytest.mark.parametrize(
    ("rules", "updates", "expected"),
    (
        (EXAMPLE_D4, "75,47,61,53,29", True),
        (EXAMPLE_D4, "75,97,47,61,53", False),
    ),
)
def test_ordering(rules, updates, expected):
    assert is_ordered(rules, updates) == expected


# AOC 2024 day 4

EXAMPLE_GRID = """
MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX
"""


class Grid:
    def __init__(self, s: str) -> None:
        self.grid: dict[tuple[int, int], str] = {}
        rows = list(filter(None, s.split("\n")))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.grid[(r, c)] = value

        self.num_rows = len(rows)
        self.num_cols = len(rows[0])
        self.deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def all_points(self) -> Iterable[tuple[int, int]]:
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                yield (r, c)

    def count_str(self, start: tuple[int, int], s: str) -> int:
        result = 0
        for delta in self.deltas:
            p = start
            aborted = False
            for c in s:
                if p not in self.grid or self.grid[p] != c:
                    aborted = True
                    break
                p = (p[0] + delta[0], p[1] + delta[1])
            if not aborted:
                result += 1
        return result


def count_xmas(s: str) -> int:
    g = Grid(s)
    return sum(g.count_str(p, "XMAS") for p in g.all_points())


@pytest.mark.parametrize(
    ("input", "expected"),
    (  # force
        ("XMAS", 1),
        (EXAMPLE_GRID, 18),
    ),
)
def test_xmas(input, expected):
    assert count_xmas(input) == expected


# AOC 2024 day 3
def parse_d3(x: str) -> int:
    regex = re.compile(r"mul\((\d+),(\d+)\)")
    result = 0
    start = 0
    while True:
        match = regex.search(x, start)
        if match:
            result += int(match.group(1)) * int(match.group(2))
            start = match.end()
        else:
            break
    return result


@pytest.mark.parametrize(
    ("input", "expected"),
    (
        ("mul(11,8)", 88),
        ("mul(3,2)uiqhaUHmul(4,5)", 26),
    ),
)
def test_parse_d3(input, expected):
    assert parse_d3(input) == expected


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
