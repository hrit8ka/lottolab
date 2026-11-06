"""Validation rules for Loto 6/40 draws."""
from __future__ import annotations
from datetime import date, datetime
from typing import Iterable

N_NUMBERS = 6
MAX_NUMBER = 40


class InvalidDraw(ValueError):
    pass


def validate_numbers(numbers: Iterable[int]) -> tuple[int, ...]:
    nums = tuple(sorted(int(n) for n in numbers))
    if len(nums) != N_NUMBERS:
        raise InvalidDraw(f"expected {N_NUMBERS} numbers, got {len(nums)}: {nums}")
    if len(set(nums)) != N_NUMBERS:
        raise InvalidDraw(f"numbers must be distinct: {nums}")
    if nums[0] < 1 or nums[-1] > MAX_NUMBER:
        raise InvalidDraw(f"numbers must be in 1..{MAX_NUMBER}: {nums}")
    return nums


def parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise InvalidDraw(f"unrecognised date format: {value!r}")


def validate_draw(draw_date, numbers) -> tuple[date, tuple[int, ...]]:
    return parse_date(draw_date), validate_numbers(numbers)
