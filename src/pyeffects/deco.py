import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from pyresults import Result

from pyeffects.effect import Effect, FnEffect

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
P = ParamSpec("P")


def effect(func: Callable[P, T]) -> Callable[P, Effect[T]]:
    """Decorator to convert a function into an effect"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Effect[T]:
        return FnEffect(lambda: func(*args, **kwargs))

    return wrapper


def io(func: Callable[P, T]) -> Callable[P, Effect[T]]:
    """Decorator to convert a function into an IO effect (alias for effect)"""
    return effect(func)


def safe(func: Callable[P, Result[T, E]]) -> Callable[P, Effect[Result[T, E]]]:
    """Decorator to convert a function into a safe effect"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Effect[Result[T, E]]:
        return FnEffect(lambda: func(*args, **kwargs))

    return wrapper
