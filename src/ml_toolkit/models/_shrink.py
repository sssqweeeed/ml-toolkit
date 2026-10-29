"""Shared validation for optional model shrinking, including serving adapters."""


def validate_size(n_iterations: int, available: int) -> None:
    if type(n_iterations) is not int or not 1 <= n_iterations <= available:
        raise ValueError(f"n_iterations must be an integer between 1 and {available}")
