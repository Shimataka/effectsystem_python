"""Effect system with Function Operations.

This module provides the implementations of the effect system.

"""

import concurrent.futures
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    ParamSpec,
    TypeVar,
)

from pyeffects.exception import FilterError

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
P = ParamSpec("P")


class Effect(Generic[T], ABC):
    """Base class of the effect system"""

    @abstractmethod
    def unwrap(self) -> T:
        """Execute the effect and return the result"""

    def map(self, func: Callable[[T], U]) -> "Effect[U]":
        """Map the effect"""
        return MappedEffect(self, func)

    def flat_map(self, func: Callable[[T], "Effect[U]"]) -> "Effect[U]":
        """Flat map the effect"""
        return FlatMappedEffect(self, func)

    def filter(self, predicate: Callable[[T], bool]) -> "Effect[T]":
        """Filter the effect"""
        return FilteredEffect(self, predicate)

    def find(self, predicate: Callable[[T], bool]) -> "Effect[T | None]":
        """Find the first element that satisfies the predicate"""
        return FilteredOptionalEffect(self, predicate)

    def take_if(self, predicate: Callable[[T], bool]) -> "Effect[T]":
        """Take the value if it satisfies the predicate"""
        return self.filter(predicate)

    def reject(self, predicate: Callable[[T], bool]) -> "Effect[T]":
        """Reject the value if it satisfies the predicate"""
        return FilteredEffect(self, lambda x: not predicate(x))

    def recover_from_none(self: "Effect[T | None]", default_factory: Callable[[], T]) -> "Effect[T]":
        """Convert Optional[T] -> T (default value if None)"""

        def recover() -> T:
            result = self.unwrap()
            return result if result is not None else default_factory()

        return FnEffect(recover)

    def or_else(self: "Effect[T | None]", alternative: "Effect[T]") -> "Effect[T]":
        """Execute the alternative effect if the value is None"""

        def or_else_action() -> T:
            result = self.unwrap()
            return result if result is not None else alternative.unwrap()

        return FnEffect(or_else_action)

    def exists(self, predicate: Callable[[T], bool]) -> "Effect[bool]":
        """Check if the value satisfies the predicate"""

        def exists_check() -> bool:
            try:
                result = self.unwrap()
                return predicate(result)
            except Exception:  # noqa: BLE001
                return False

        return FnEffect(exists_check)

    def zip_with(self, other: "Effect[U]", combiner: Callable[[T, U], Any]) -> "Effect[Any]":
        """Zip the effect with another effect"""
        return ZippedEffect(self, other, combiner)

    def tap(self, side_effect: Callable[[T], None]) -> "Effect[T]":
        """Tap effect (side effect)"""
        return TappedEffect(self, side_effect)

    def recover(self, handler: Callable[[Exception], T]) -> "Effect[T]":
        """Recover the effect"""
        return RecoveredEffect(self, handler)

    def retry(self, max_retries: int = 3, delay: float = 0.1) -> "Effect[T]":
        """Retry the effect"""
        return RetriedEffect(self, max_retries, delay)

    def memorize(self) -> "Effect[T]":
        """Memorize the effect"""
        return MemorizedEffect(self)

    def timeout(self, seconds: float) -> "Effect[T]":
        """Timeout the effect"""
        return TimedOutEffect(self, seconds)

    @classmethod
    def pure(cls, value: T) -> "Effect[T]":
        """Create a pure effect"""
        return PureEffect(value)

    @classmethod
    def sequence(cls, effects: list["Effect[T]"]) -> "Effect[list[T]]":
        """Sequence the effects"""
        return SequenceEffect(effects)

    @classmethod
    def parallel(cls, effects: list["Effect[T]"], max_workers: int | None = None) -> "Effect[list[T]]":
        """Parallel the effects"""
        return ParallelEffect(effects, max_workers)

    @classmethod
    def race(cls, effects: list["Effect[T]"]) -> "Effect[T]":
        """Race the effects"""
        return RaceEffect(effects)


@dataclasses.dataclass(slots=True)
class FnEffect(Effect[T]):
    """Effect that is a function"""

    func: Callable[[], T]

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        return self.func()


@dataclasses.dataclass(slots=True)
class PureEffect(Effect[T]):
    """Effect that is a pure value"""

    value: T

    def unwrap(self) -> T:
        """Return the value"""
        return self.value


@dataclasses.dataclass(slots=True)
class MappedEffect(Generic[T, U], Effect[U]):
    """Effect that is a mapped value"""

    source: Effect[T]
    mapper: Callable[[T], U]

    def unwrap(self) -> U:
        """Execute the effect and return the result"""
        return self.mapper(self.source.unwrap())


@dataclasses.dataclass(slots=True)
class FlatMappedEffect(Generic[T, U], Effect[U]):
    """Effect that is a flat mapped value"""

    source: Effect[T]
    mapper: Callable[[T], Effect[U]]

    def unwrap(self) -> U:
        """Execute the effect and return the result"""
        return self.mapper(self.source.unwrap()).unwrap()


@dataclasses.dataclass(slots=True)
class FilteredEffect(Generic[T], Effect[T]):
    """Effect that is a filtered value"""

    source: Effect[T]
    predicate: Callable[[T], bool]

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        result = self.source.unwrap()
        if self.predicate(result):
            return result
        raise FilterError(result)


@dataclasses.dataclass(slots=True)
class FilteredOptionalEffect(Generic[T], Effect[T | None]):
    """Effect that is a filtered value"""

    source: Effect[T]
    predicate: Callable[[T], bool]

    def unwrap(self) -> T | None:
        """Execute the effect and return the result"""
        result = self.source.unwrap()
        return result if self.predicate(result) else None


@dataclasses.dataclass(slots=True)
class ZippedEffect(Generic[T, U], Effect[Any]):
    """Effect that is a zipped value"""

    left: Effect[T]
    right: Effect[U]
    combiner: Callable[[T, U], Any]

    def unwrap(self) -> Any:  # noqa: ANN401
        """Execute the effect and return the result"""
        return self.combiner(self.left.unwrap(), self.right.unwrap())


@dataclasses.dataclass(slots=True)
class TappedEffect(Effect[T]):
    """Effect that is a tapped value"""

    source: Effect[T]
    side_effect: Callable[[T], None]

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        result = self.source.unwrap()
        self.side_effect(result)
        return result


@dataclasses.dataclass(slots=True)
class RecoveredEffect(Effect[T]):
    """Effect that is a recovered value"""

    source: Effect[T]
    handler: Callable[[Exception], T]

    def unwrap(self) -> T:
        try:
            return self.source.unwrap()
        except Exception as e:  # noqa: BLE001
            return self.handler(e)


@dataclasses.dataclass(slots=True)
class RetriedEffect(Effect[T]):
    """Effect that is a retried value"""

    source: Effect[T]
    max_retries: int
    delay: float

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        import time

        last_exception = Exception("Max retries reached")

        for i in range(self.max_retries):
            try:
                return self.source.unwrap()
            except Exception as e:
                last_exception = e
                if i < self.max_retries - 1:
                    time.sleep(self.delay)
                    self.delay *= 2
                else:
                    raise

        raise last_exception


@dataclasses.dataclass(slots=True)
class MemorizedEffect(Effect[T]):
    """Effect that is a memorized value"""

    source: Effect[T]
    _cached_result: T | None = dataclasses.field(default=None, init=False)
    _computed: bool = dataclasses.field(default=False, init=False)

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        if not self._computed:
            self._cached_result = self.source.unwrap()
            self._computed = True
        return self._cached_result  # type: ignore[return-value]


@dataclasses.dataclass(slots=True)
class TimedOutEffect(Effect[T]):
    """Effect that is a timed out value"""

    source: Effect[T]
    seconds: float

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future: concurrent.futures.Future[T] = executor.submit(self.source.unwrap)
            try:
                return future.result(timeout=self.seconds)
            except concurrent.futures.TimeoutError as e:
                future.cancel()
                msg = f"Effect timed out after {self.seconds} seconds"
                raise TimeoutError(msg) from e


@dataclasses.dataclass(slots=True)
class SequenceEffect(Effect[list[T]]):
    """Effect that is a sequence of effects"""

    effects: list[Effect[T]]

    def unwrap(self) -> list[T]:
        """Execute the effect and return the result"""
        return [effect.unwrap() for effect in self.effects]


@dataclasses.dataclass(slots=True)
class ParallelEffect(Effect[list[T]]):
    """Effect that is a parallel of effects"""

    effects: list[Effect[T]]
    max_workers: int | None = None

    def unwrap(self) -> list[T]:
        """Execute the effect and return the result"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: list[concurrent.futures.Future[T]] = [executor.submit(effect.unwrap) for effect in self.effects]
            return [future.result() for future in futures]


@dataclasses.dataclass(slots=True)
class RaceEffect(Effect[T]):
    """Effect that is a race of effects"""

    effects: list[Effect[T]]

    def unwrap(self) -> T:
        """Execute the effect and return the result"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures: list[concurrent.futures.Future[T]] = [executor.submit(effect.unwrap) for effect in self.effects]
            try:
                done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in not_done:
                    future.cancel()
                return next(iter(done)).result()
            except Exception:
                for future in futures:
                    future.cancel()
                raise
