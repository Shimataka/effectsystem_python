from collections.abc import Callable
from typing import TypeVar

from .base import FnEffect, PureEffect

T = TypeVar("T")


class Eff(PureEffect[T]):
    """Alias for PureEffect"""

    def __init__(
        self,
        value: T,
    ) -> None:
        super().__init__(value)


class FnEff(FnEffect[T]):
    """Alias for FnEffect"""

    def __init__(
        self,
        func: Callable[[], T],
    ) -> None:
        super().__init__(func)
