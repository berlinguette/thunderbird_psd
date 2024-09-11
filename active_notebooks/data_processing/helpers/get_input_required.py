from typing import Callable, TypeVar

T = TypeVar('T')


def get_input_required(
    prompt: str, allowable_values: list[T] | tuple[T | None, T | None], converter: Callable[[str], T]
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
        if isinstance(allowable_values, list):
            if value not in allowable_values:
                print("The given value is not one of the allowable options.  Please try again.")
                continue
        else:
            lo_limit, hi_limit = allowable_values
            try:
                if lo_limit is not None and value < lo_limit:
                    print("The given value is lower than the lower allowable limit.  Please try again.")
                    continue
                if hi_limit is not None and value > hi_limit:
                    print("The given value is higher than the higher allowable limit.  Please try again.")
                    continue
            except TypeError as err:
                print("Allowable limits are not supported for the given data type")
                raise err
        done = True
    return value
