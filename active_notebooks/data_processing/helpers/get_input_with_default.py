from typing import Callable, TypeVar

T = TypeVar('T')


def get_input_with_default(
    prompt: str, default: T, converter: Callable[[str], T]
) -> T:
    raw_value = input(prompt)
    if raw_value == "":
        return default
    try:
        value = converter(raw_value)
    except ValueError:
        value = default
    return value
