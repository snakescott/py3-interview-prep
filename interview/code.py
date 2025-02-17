import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import more_itertools
import pytest
from boltons.queueutils import PriorityQueue  # type: ignore[import-untyped]

# AOC 2024 day 10
# seems like a graph search problem, constraints on grid edges
# one option is to propogate the number of paths ending at 9


@dataclass
@dataclass
class DayTenValue:
    elevation: int
    peaks: set[int]

    def __init__(self, elevation, location):
        self.elevation = elevation
        self.peaks = set()
        if elevation == 9:
            self.peaks.add(location)


class Day10Graph(defaultdict[complex, DayTenValue]):
    pass


DAY_10_DELTAS = [-1, 1, -1j, 1j]


def day10_parse(s: str) -> Day10Graph:
    g = Day10Graph()
    rows = s.split("\n")
    for y, row in enumerate(rows):
        for x, elevation in enumerate(row):
            location = x + 1j * y
            g[location] = DayTenValue(elevation=int(elevation), location=location)
    return g


def trailhead_scores(g: Day10Graph) -> list[int]:
    by_elevation: dict[int, Day10Graph] = defaultdict(lambda: Day10Graph())
    for k, v in g.items():
        by_elevation[v.elevation][k] = v

    for elevation in range(9, 0, -1):
        current_level = by_elevation[elevation]
        descent_level = by_elevation[elevation - 1]
        for position, value in current_level.items():
            for delta in DAY_10_DELTAS:
                if position + delta in descent_level:
                    descent_level[position + delta].peaks |= value.peaks
    return [len(v.peaks) for v in by_elevation[0].values()]


@dataclass
@dataclass
class DayTenBValue:
    elevation: int
    paths: int

    def __init__(self, elevation, location):
        self.elevation = elevation
        self.paths = 0
        if elevation == 9:
            self.paths = 1


class Day10BGraph(defaultdict[complex, DayTenBValue]):
    pass


def day10b_parse(s: str) -> Day10BGraph:
    g = Day10BGraph()
    rows = s.split("\n")
    for y, row in enumerate(rows):
        for x, elevation in enumerate(row):
            location = x + 1j * y
            g[location] = DayTenBValue(elevation=int(elevation), location=location)
    return g


def trailhead_b_scores(g: Day10BGraph) -> list[int]:
    by_elevation: dict[int, Day10BGraph] = defaultdict(lambda: Day10BGraph())
    for k, v in g.items():
        by_elevation[v.elevation][k] = v

    for elevation in range(9, 0, -1):
        current_level = by_elevation[elevation]
        descent_level = by_elevation[elevation - 1]
        for position, value in current_level.items():
            for delta in DAY_10_DELTAS:
                if position + delta in descent_level:
                    descent_level[position + delta].paths += value.paths
    return [v.paths for v in by_elevation[0].values()]


DAY10_EXAMPLE = """89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732"""


@pytest.mark.parametrize(
    ("s", "expected"),
    (  # format
        (DAY10_EXAMPLE, 36),
    ),
)
def test_trails(s, expected):
    g = day10_parse(s)
    assert sum(trailhead_scores(g)) == expected


def test_2024_d10_1():
    data = load_input(2024, 10)
    g = day10b_parse(data)
    assert sum(trailhead_b_scores(g)) == 489


def test_2024_d10_2():
    data = load_input(2024, 10)
    g = day10_parse(data)
    assert sum(trailhead_scores(g)) == 1086


# AOC 2024 day 7
EXAMPLE_D7 = """190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20"""


@dataclass
class DaySevenEntry:
    target: int
    values: list[int]


def parse_d7(input: str) -> list[DaySevenEntry]:
    def parse_line(line):
        x, y = line.split(": ")
        return DaySevenEntry(target=int(x), values=list(map(int, y.split(" "))))

    return [parse_line(l) for l in input.split("\n")]


def can_make(target: int, values: list[int], index: int, concat: bool = False) -> bool:
    if index == len(values) - 1:
        return values[index] == target

    old = values[index + 1]
    candidates = [old + values[index], old * values[index]]

    if concat:
        mult = 10 ** (len(str(old)))
        candidates.append(mult * values[index] + old)

    for candidate in candidates:
        values[index + 1] = candidate
        if can_make(target, values, index + 1, concat=concat):
            values[index + 1] = old
            return True

    values[index + 1] = old
    return False


@pytest.mark.parametrize(
    ("s", "expected"),
    (  # format
        ("190: 10 19", True),
        ("83: 17 5", False),
    ),
)
def test_can_make(s, expected):
    n = parse_d7(s)[0]
    assert can_make(n.target, n.values, 0) == expected


@pytest.mark.parametrize(
    ("s", "concat", "expected"),
    (  # format
        (EXAMPLE_D7, False, 3749),
        (EXAMPLE_D7, True, 11387),
        (EXAMPLE_D7.split("\n")[4], True, 7290),
    ),
)
def test_2024_d7_misc(s, concat, expected):
    nodes = parse_d7(s)
    assert sum(n.target for n in nodes if can_make(n.target, n.values, 0, concat=concat)) == expected


def test_2024_d7_1():
    data = load_input(2024, 7)
    nodes = parse_d7(data)
    assert sum(n.target for n in nodes if can_make(n.target, n.values, 0)) == 6083020304036


# Slow
# def test_2024_d7_2():
#     data = load_input(2024, 7)
#     nodes = parse_d7(data)
#     assert sum(n.target for n in nodes if can_make(n.target, n.values, 0, concat=True)) == 59002246504791


# AOC 2024 day 6

EXAMPLE_D6 = """....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#..."""


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
53|13"""


def is_ordered(rules: list[list[int]], updates: list[int]):
    not_after: dict[int, set[int]] = defaultdict(lambda: set())
    for a, b in rules:
        not_after[a].add(b)

    seen: set[int] = set()
    for update in updates:
        if seen.intersection(not_after[update]):
            return False
        seen.add(update)
    return True


@dataclass
class DayFiveNode:
    value: int
    index: int
    before: set[int] = field(default_factory=set)
    after: set[int] = field(default_factory=set)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return hash((self.value, self.index))


def reorder(rules: list[list[int]], updates: list[int]):
    nodes: dict[int, DayFiveNode] = {}

    for index, value in enumerate(updates):
        n = DayFiveNode(value=value, index=index)
        nodes[value] = n

    for src, dst in rules:
        if src in nodes and dst in nodes:
            nodes[dst].before.add(src)
            nodes[src].after.add(dst)

    result: list[int] = []
    pq = PriorityQueue()
    for n in nodes.values():
        if not n.before:
            pq.add((n.index, n))

    while pq:
        target = pq.pop()[1]
        result.append(target.value)
        for value in target.after:
            node = nodes[value]
            node.before.remove(target.value)
            if not node.before:
                pq.add((node.index, node))

    return result


@pytest.mark.parametrize(
    ("raw_rules", "raw_updates", "expected"),
    (("1|2", "2,1", [1, 2]),),
)
def test_reordering(raw_rules, raw_updates, expected):
    rules = [list(map(int, l.split("|"))) for l in raw_rules.split("\n")]
    updates = list(map(int, raw_updates.split(",")))
    assert reorder(rules, updates) == expected


@pytest.mark.parametrize(
    ("raw_rules", "raw_updates", "expected"),
    (
        (EXAMPLE_D4, "75,47,61,53,29", True),
        (EXAMPLE_D4, "75,97,47,61,53", False),
    ),
)
def test_ordering(raw_rules, raw_updates, expected):
    rules = [list(map(int, l.split("|"))) for l in raw_rules.split("\n")]
    updates = list(map(int, raw_updates.split(",")))
    assert is_ordered(rules, updates) == expected


def compute_d5_from_raw(s: str) -> int:
    raw_rules, raw_updates = s.split("\n\n")
    rules = [list(map(int, l.split("|"))) for l in raw_rules.split("\n")]
    updates = [list(map(int, raw_update.split(","))) for raw_update in raw_updates.split("\n")]
    return sum(u[int(len(u) / 2)] for u in updates if is_ordered(rules, u))


def test_2024_d5_1():
    data = load_input(2024, 5)
    assert compute_d5_from_raw(data) == 6612


def test_2024_d5_2():
    data = load_input(2024, 5)
    raw_rules, raw_updates = data.split("\n\n")
    rules = [list(map(int, l.split("|"))) for l in raw_rules.split("\n")]
    updates = [list(map(int, raw_update.split(","))) for raw_update in raw_updates.split("\n")]
    bad_updates = [reorder(rules, u) for u in updates if not is_ordered(rules, u)]
    assert sum(u[int(len(u) / 2)] for u in bad_updates) == 4944


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
        self.grid: dict[complex, str] = {}
        rows = list(filter(None, s.split("\n")))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.grid[r + c * 1j] = value

        self.num_rows = len(rows)
        self.num_cols = len(rows[0])
        self.deltas = [-1 - 1j, -1, -1 + 1j, -1j, 1j, 1 - 1j, 1, 1 + 1j]

    def all_points(self) -> Iterable[complex]:
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                yield r + c * 1j

    def count_str(self, start: complex, s: str) -> int:
        result = 0
        for delta in self.deltas:
            p = start
            aborted = False
            for c in s:
                if p not in self.grid or self.grid[p] != c:
                    aborted = True
                    break
                p = p + delta
            if not aborted:
                result += 1
        return result

    def is_x_mas(self, start: complex) -> bool:
        if self.grid.get(start) != "A":
            return False

        diag_a = set([self.grid.get(start - 1 - 1j), self.grid.get(start + 1 + 1j)])
        diag_b = set([self.grid.get(start + 1 - 1j), self.grid.get(start - 1 + 1j)])
        sm = set(["S", "M"])
        return diag_a == sm and diag_b == sm


def count_xmas(s: str) -> int:
    g = Grid(s)
    return sum(g.count_str(p, "XMAS") for p in g.all_points())


def count_x_mas(s: str) -> int:
    g = Grid(s)
    return sum(1 if g.is_x_mas(p) else 0 for p in g.all_points())


def test_2024_d4_1():
    data = load_input(2024, 4)
    assert count_xmas(data) == 2593


def test_2024_d4_2():
    data = load_input(2024, 4)
    assert count_x_mas(data) == 1950


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


def parse_d3_do_dont(x: str) -> int:
    parts = x.split("don't()")
    if parts:
        value = parse_d3(parts[0])
        for p in parts[1:]:
            start = p.find("do()")
            if start > 0:
                value += parse_d3(p[start:])
        return value
    return 0


@pytest.mark.parametrize(
    ("input", "expected"),
    (
        ("mul(11,8)", 88),
        ("mul(3,2)uiqhaUHmul(4,5)", 26),
    ),
)
def test_parse_d3(input, expected):
    assert parse_d3(input) == expected


def parse_2024_d3():
    data = load_input(2024, 3)
    return data or ""


def test_2024_d3_1():
    assert parse_d3(parse_2024_d3()) == 155955228


def test_2024_d3_2():
    data = "".join(parse_2024_d3())
    assert parse_d3_do_dont(parse_2024_d3()) == 100189366


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


def parse_2024_d2():
    data = load_input(2024, 2)
    if not data:
        return []
    return [list(map(int, line.split())) for line in data.split("\n")]


# Can do in linear time by stitching partial sequences from start and back
def is_damper_safe(l: list[int]) -> bool:
    if is_safe(l):
        return True
    for idx in range(len(l)):
        l2 = list(l)
        del l2[idx]
        if is_safe(l2):
            return True

    return False


def test_2024_d2_1():
    l = parse_2024_d2()
    assert sum(1 if is_safe(x) else 0 for x in l) == 502


def test_2024_d2_2():
    l = parse_2024_d2()
    assert sum(1 if is_damper_safe(x) else 0 for x in l) == 544


# AOC 2024 day 1
def distance(a: list[int], b: list[int]) -> int:
    a = sorted(a)
    b = sorted(b)
    return sum(abs(x[0] - x[1]) for x in zip(a, b, strict=True))


def similarity(a: list[int], b: list[int]) -> int:
    c = Counter(b)
    return sum(i * c[i] for i in a)


def parse_2024_d1():
    data = load_input(2024, 1)
    if not data:
        return [], []
    lines = data.split("\n")
    a = []
    b = []
    regex = re.compile(r"(\d+) +(\d+)")
    for line in lines:
        match = regex.search(line)
        a.append(int(match.group(1)))
        b.append(int(match.group(2)))
    return a, b


def test_2024_d1_2():
    a, b = parse_2024_d1()
    assert similarity(a, b) == 26674158


def test_2024_d1_1():
    a, b = parse_2024_d1()
    assert distance(a, b) == 1830467


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


def load_input(year: int, day: int) -> str | None:
    """
    Load the contents of an Advent of Code input file for the given year and day.

    Args:
        year (int): The year of the puzzle
        day (int): The day of the puzzle

    Returns:
        str | None: The contents of the input file if it exists, None otherwise
    """
    # Get the directory containing the current file
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Construct the path to the input file
    input_path = current_dir / "../aoc_inputs" / f"{year}_d{day}.input"
    print(input_path)
    # Check if the file exists and return its contents if it does
    if input_path.exists():
        return input_path.read_text().strip()

    return None


# Allows invocation as a main file, e.g. if using inside coderpad or similar
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
