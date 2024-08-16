from typing import Callable, TypeVar

T = TypeVar('T')


def get_input_with_default(
    prompt: str, default: T, converter: Callable[[str], T]
) -> T:
    """Prompts the user for an input value of a specific data type
    The user can press Enter without inputting a value to get a default value.
    If the user enters a value that cannot be converted to the desired data type, 
    this function will retry until the entered value can be converted.

    :param prompt: Input prompt shown to the user
    :type prompt: str
    :param default: Default value
    :type default: T
    :param converter: Function that converts the string input to the desired data type
    :type converter: Callable[[str], T]
    :return: Input result in the desired data type, or the default value
    :rtype: T
    """
    raw_value = input(prompt)
    if raw_value == "":
        return default
    try:
        value = converter(raw_value)
    except ValueError:
        value = default
    return value
