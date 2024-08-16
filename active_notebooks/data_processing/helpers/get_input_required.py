from typing import Callable, TypeVar

T = TypeVar('T')


def get_input_required(
    prompt: str, allowable_values: list[T], converter: Callable[[str], T]
) -> T:
    """Prompts the user for an input value of a specific data type
    This function requires that the user enter a valid value, and will retry until this is done.

    :param prompt: Input prompt shown to the user
    :type prompt: str
    :param allowable_values: List of allowable values. If the given value is not in this list, it is not accepted
    :type allowable_values: list[T]
    :param converter: Function that converts the string input to the desired data type
    :type converter: Callable[[str], T]
    :return: Input result in the desired data type
    :rtype: T
    """
    while True:
        raw_value = input(prompt)
        try:
            value = converter(raw_value)
        except ValueError:
            print("The given value could not be properly converted.  Please try again.")
            continue
        if value not in allowable_values:
            print("The given value is not one of the allowable options.  Please try again.")
            continue
        return value
