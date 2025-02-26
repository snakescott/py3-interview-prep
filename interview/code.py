import os
import re
from collections import Counter, defaultdict, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from pathlib import Path

import more_itertools
import pytest
from boltons.queueutils import PriorityQueue  # type: ignore[import-untyped]

EIGHT_DELTA = [-1 - 1j, -1, -1 + 1j, -1j, 1j, 1 - 1j, 1, 1 + 1j]


# AOC 2023 day 5
Day5Range = namedtuple("Day5Range", ["start", "length", "delta"])

DAY5_KEY_SEQUENCE = [
    "seed-to-soil",
    "soil-to-fertilizer",
    "fertilizer-to-water",
    "water-to-light",
    "light-to-temperature",
    "temperature-to-humidity",
    "humidity-to-location",
]

DAY5_EXAMPLE1 = """seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4"""


def parse_2023d5(s: str) -> tuple[list[int], dict[str, list[Day5Range]]]:
    seed_line, _, rest = s.partition("\n\n")
    seed_line = seed_line.partition("seeds: ")[2]
    seeds = [int(part) for part in seed_line.split()]
    maps: dict[str, list[Day5Range]] = {}
    raw_maps = rest.split("\n\n")
    max_length = -1
    for m in raw_maps:
        first_line, _, rest = m.partition("\n")
        lines = rest.split("\n")

        key = first_line.split(" ")[0]
        ranges: list[Day5Range] = []
        for line in lines:
            parts = line.split()
            dest, start, length = int(parts[0]), int(parts[1]), int(parts[2])
            ranges.append(Day5Range(start, length, dest - start))

        maps[key] = sorted(ranges)
    return (seeds, maps)


@dataclass
class Day5Almanac:
    seeds: list[int]
    maps: dict[str, list[Day5Range]]

    def seed_to_location(self, seed: int):
        current = seed
        for key in DAY5_KEY_SEQUENCE:
            ranges = self.maps[key]
            # Can do log(n) with binary search
            for idx in range(len(ranges) - 1, -1, -1):
                d5range = ranges[idx]
                if current >= d5range.start:
                    if current < d5range.start + ranges[idx].length:
                        current += d5range.delta
                    break
        return current

    @staticmethod
    def parse(s: str) -> "Day5Almanac":
        seeds, maps = parse_2023d5(s)
        return Day5Almanac(seeds=seeds, maps=maps)


# PART TWO RANGE OPERATIONS


@pytest.mark.parametrize(
    ("input", "seeds", "locations"),
    (  # format
        (DAY5_EXAMPLE1, [79, 14, 55, 13], [82, 43, 86, 35]),
    ),
)
def test_day5_misc(input, seeds, locations):
    almanac = Day5Almanac.parse(input)
    for s, l in zip(seeds, locations, strict=False):
        assert almanac.seed_to_location(s) == l


def test_2023_d5_1():
    input = load_input(2023, 5)
    almanac = Day5Almanac.parse(input)
    seeds = set(almanac.seeds)
    locations = set(almanac.seed_to_location(s) for s in seeds)
    target = min(locations)
    assert target == 621354867


def test_2023_d5_2():
    pass


# AOC 2023 day 4
@dataclass
class Day4Card:
    id_: int
    picks: set[int]
    winners: set[int]

    # # parse e.g. "Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53"
    @staticmethod
    def parse(s: str) -> "Day4Card":
        parts = s.split(":")
        id_ = int(parts[0].split()[-1])
        raw_picks, raw_winners = parts[1].split("|")
        picks = set(map(int, raw_picks.split()))
        winners = set(map(int, raw_winners.split()))
        return Day4Card(id_=id_, picks=picks, winners=winners)

    def score(self) -> int:
        shared = self.picks.intersection(self.winners)
        if len(shared) == 0:
            return 0
        else:
            return 2 ** (len(shared) - 1)

    def lucky(self):
        return len(self.picks.intersection(self.winners))

    @staticmethod
    def total(cards: "list[Day4Card]") -> int:
        counts = [1 for i in range(0, len(cards))]
        for idx in range(0, len(cards)):
            count = counts[idx]
            card = cards[idx]
            for i in range(1, card.lucky() + 1):
                inc_idx = idx + i
                if inc_idx < len(counts):
                    counts[inc_idx] += count
        return sum(counts)


def test_2023_d4_1():
    input = load_input(2023, 4)
    rows = input.split("\n")
    cards = [Day4Card.parse(r) for r in rows]
    assert sum(c.score() for c in cards) == 19135


def test_2023_d4_2():
    input = load_input(2023, 4)
    rows = input.split("\n")
    cards = [Day4Card.parse(r) for r in rows]
    assert Day4Card.total(cards) == 5704953


# AOC 2023 day 3
# Somewhat complex, need to build integers out of strings but also find adjacent symbols
# Approach: parse numbers with metadata, parse symbols, check

Day3_1_EXAMPLE = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598.."""


@dataclass
class Day3Number:
    value: int
    row: int
    start_col: int
    end_col: int

    @staticmethod
    def parse(row: int, s: str, start: int) -> "Day3Number | None":
        s = s + "."
        start_col: int | None = None
        value = 0
        i = 0
        for i in range(start, len(s)):
            ch = s[i]
            is_number = ch >= "0" and ch <= "9"
            if is_number and start_col is not None:
                value = value * 10 + int(ch)
            elif start_col is not None:
                break
            elif is_number:
                value = int(ch)
                start_col = i

        if start_col is not None:
            return Day3Number(value=value, row=row, start_col=start_col, end_col=i)
        return None

    @staticmethod
    def parse_all(rows: list[str]) -> "list[Day3Number]":
        numbers: list[Day3Number] = []
        for rid, row in enumerate(rows):
            cid = 0
            _next = Day3Number.parse(rid, row, cid)
            while _next is not None:
                numbers.append(_next)
                cid = _next.end_col
                _next = Day3Number.parse(rid, row, cid)

        return numbers


def find_symbols(schematic: list[str]) -> set[complex]:
    result: set[complex] = set()
    for rid, row in enumerate(schematic):
        for cid, ch in enumerate(row):
            if (ch < "0" or ch > "9") and ch != ".":
                result.add(rid + 1j * cid)
    return result


def is_part_number(n: Day3Number, symbols: set[complex]) -> bool:
    for cid in range(n.start_col, n.end_col):
        pos = n.row + 1j * cid
        for delta in EIGHT_DELTA:
            if pos + delta in symbols:
                return True
    return False


def adjacent_symbols(n: Day3Number, symbols: set[complex]) -> set[complex]:
    result: set[complex] = set()
    for cid in range(n.start_col, n.end_col):
        pos = n.row + 1j * cid
        for delta in EIGHT_DELTA:
            if pos + delta in symbols:
                result.add(pos + delta)
    return result


def all_adjacent_part_numbers(numbers: list[Day3Number], symbols: set[complex]) -> dict[complex, list[Day3Number]]:
    result: defaultdict[complex, list[Day3Number]] = defaultdict(lambda: [])
    for n in numbers:
        for symbol in adjacent_symbols(n, symbols):
            result[symbol].append(n)
    return result


@pytest.mark.parametrize(
    ("input", "expected"),
    (  # format
        (Day3_1_EXAMPLE, 4361),
    ),
)
def test_day3_misc(input, expected):
    rows = input.split("\n")
    numbers = Day3Number.parse_all(rows)
    symbols = find_symbols(rows)
    assert sum(n.value for n in numbers if is_part_number(n, symbols)) == expected


def test_2023_d3_1():
    input = load_input(2023, 3)
    rows = input.split("\n")
    numbers = Day3Number.parse_all(rows)
    symbols = find_symbols(rows)
    assert sum(n.value for n in numbers if is_part_number(n, symbols)) == 527364


def test_2023_d3_1():
    input = load_input(2023, 3)
    rows = input.split("\n")
    numbers = Day3Number.parse_all(rows)
    symbols = find_symbols(rows)
    misc = all_adjacent_part_numbers(numbers, symbols)
    value = 0
    for symbol, numbers in misc.items():
        if len(numbers) == 2:
            value += numbers[0].value * numbers[1].value
    assert value == 79026871


# AOC 2023 day 2
# going to be about parsing and data representation mostly


@dataclass
class Day2Reveal:
    groups: dict[str, int]

    @staticmethod
    def parse(s: str) -> "Day2Reveal":
        cubes = s.split(", ")
        groups: dict[str, int] = defaultdict(lambda: 0)
        for c in cubes:
            parts = c.split(" ")
            count = int(parts[0])
            color = parts[1]
            groups[color] = count
        return Day2Reveal(groups=groups)

    def power(self):
        return reduce(mul, [self.groups["blue"], self.groups["green"], self.groups["red"]])


@dataclass
class Day2Game:
    game_id: int
    reveals: list[Day2Reveal]

    @staticmethod
    def parse(s: str) -> "Day2Game":
        parts = s.split(": ")
        game_id = int(parts[0].split(" ")[1])
        reveals = parts[1].split("; ")
        return Day2Game(game_id=game_id, reveals=[Day2Reveal.parse(r) for r in reveals])

    def possible(self, cubes: Day2Reveal) -> bool:
        return all(all(r.groups[color] <= cubes.groups[color] for color in cubes.groups.keys()) for r in self.reveals)

    def fewest(self) -> Day2Reveal:
        groups: dict[str, int] = defaultdict(lambda: 0)
        for r in self.reveals:
            for color, count in r.groups.items():
                groups[color] = max(groups[color], count)
        return Day2Reveal(groups)


@pytest.mark.parametrize(
    ("input", "expected"),
    (  # format
        ("Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green", True),
        ("Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red", False),
    ),
)
def test_day2_game(input, expected):
    cubes = Day2Reveal({"red": 12, "green": 13, "blue": 14})
    game = Day2Game.parse(input)
    assert game.possible(cubes) == expected


def test_2023_d2_1():
    data = load_input(2023, 2)
    cubes = Day2Reveal({"red": 12, "green": 13, "blue": 14})
    games = [Day2Game.parse(l) for l in data.split("\n")]
    result = sum(game.game_id for game in games if game.possible(cubes))
    assert result == 2268


def test_2023_d2_2():
    data = load_input(2023, 2)
    games = [Day2Game.parse(l) for l in data.split("\n")]
    result = sum(game.fewest().power() for game in games)
    assert result == 63542


# AOC 2023 day 1


def calibration_value(s: str) -> int:
    def _first_digit(idx_it):
        for idx in idx_it:
            if s[idx] >= "0" and s[idx] <= "9":
                return int(s[idx])
        raise Exception("foo")

    return _first_digit(range(0, len(s))) * 10 + _first_digit(range(len(s) - 1, -1, -1))


ENGLISH = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

ENGLISH_BACKWARDS = {"".join(reversed(k)): v for k, v in ENGLISH.items()}


def calibration_value2(s: str) -> int:
    def _first_digit(inner: str, english: dict[str, int]):
        for idx in range(len(inner)):
            if inner[idx] >= "0" and inner[idx] <= "9":
                return int(inner[idx])
            for k, v in english.items():
                # slice makes a copy, RIP
                if inner[idx:].startswith(k):
                    return v
        raise Exception("foo")

    return _first_digit(s, ENGLISH) * 10 + _first_digit("".join(reversed(s)), ENGLISH_BACKWARDS)


@pytest.mark.parametrize(
    ("input", "expected"),
    (  # format
        ("1abc2", 12),
        ("pqr3stu8vwx", 38),
    ),
)
def test_calibration_value(input, expected):
    assert calibration_value(input) == expected


def test_2023_d1_1():
    data = load_input(2023, 1)
    assert (sum(calibration_value(s) for s in data.split("\n"))) == 54708


def test_2023_d1_2():
    data = load_input(2023, 1)
    assert (sum(calibration_value2(s) for s in data.split("\n"))) == 54087


# AOC 2024 day 10
# seems like a graph search problem, constraints on grid edges
# one option is to propogate the number of paths ending at 9


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


def test_2024_d10_2():
    data = load_input(2024, 10)
    g = day10b_parse(data)
    assert sum(trailhead_b_scores(g)) == 1086


def test_2024_d10_1():
    data = load_input(2024, 10)
    g = day10_parse(data)
    assert sum(trailhead_scores(g)) == 489


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
