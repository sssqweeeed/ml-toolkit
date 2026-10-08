from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def pyfunc(func: Callable[P, R]) -> Callable[P, R]: ...
