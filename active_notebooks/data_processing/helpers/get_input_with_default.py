from typing import Callable, TypeVar

T = TypeVar('T')


def get_input_with_default(
    prompt: str, default: int, converter: Callable[[str], T]
) -> int:
    raw_value = input(prompt)
    try:
        value = converter(raw_value)
    except ValueError:
        value = default
    return value
