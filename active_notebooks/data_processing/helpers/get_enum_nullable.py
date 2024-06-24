from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def get_enum_nullable(enum_value: T, enum_converter: Callable[[T], E]) -> E | None:
    try:
        return enum_converter(enum_value)
    except ValueError:
        return None
