import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from pyresults import Result

from pyeffects.base import Effect, FnEffect

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
P = ParamSpec("P")


def effect(func: Callable[P, T]) -> Callable[P, Effect[T]]:
    """Decorator to convert a function into an effect

    Args:
        func (Callable[P, T]): The function to convert into an effect.

    Returns:
        Callable[P, Effect[T]]: The decorator function.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Effect[T]:
        return FnEffect(lambda: func(*args, **kwargs))

    return wrapper


def io(func: Callable[P, T]) -> Callable[P, Effect[T]]:
    """Decorator to convert a function into an IO effect (alias for effect)

    Args:
        func (Callable[P, T]): The function to convert into an IO effect.

    Returns:
        Callable[P, Effect[T]]: The decorator function.
    """
    return effect(func)


def safe(func: Callable[P, Result[T, E]]) -> Callable[P, Effect[Result[T, E]]]:
    """Decorator to convert a function into a safe effect

    Args:
        func (Callable[P, Result[T, E]]): The function to convert into a safe effect.

    Returns:
        Callable[P, Effect[Result[T, E]]]: The decorator function.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Effect[Result[T, E]]:
        return FnEffect(lambda: func(*args, **kwargs))

    return wrapper
