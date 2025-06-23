from typing import TypeVar

from .effect import PureEffect

T = TypeVar("T")


class Eff(PureEffect[T]):
    """Alias for PureEffect"""

    def __init__(
        self,
        value: T,
    ) -> None:
        super().__init__(value)
