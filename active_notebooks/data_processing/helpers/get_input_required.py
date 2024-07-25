from typing import Callable, TypeVar

T = TypeVar('T')


def get_input_required(
    prompt: str, allowable_values: list[T], converter: Callable[[str], T]
) -> T:
    done = False
    value = None
    while not done:
        raw_value = input(prompt)
        try:
            value = converter(raw_value)
        except ValueError:
            print("The given value could not be properly converted.  Please try again.")
            continue
        if value not in allowable_values:
            print("The given value is not one of the allowable options.  Please try again.")
            continue
        done = True
    return value
