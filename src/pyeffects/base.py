"""Effect system with Function Operations.

This module provides the implementations of the effect system.

"""

import concurrent.futures
import dataclasses
import time
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
    """Base class of the effect system

    A type representing some effects that can be executed to produce a result.

    Attribures:
        - T: The type of the result of the effect.

    Notes:
        All methods are defined in the base class, and are implemented in the subclasses.

        The subclasses are:

        - FnEffect: Effect that is a function.
        - PureEffect: Effect that is a pure value.
        - MappedEffect: Effect that is a mapped value.
        - FlatMappedEffect: Effect that is a flat mapped value.
        - FilteredEffect: Effect that is a filtered value.
        - FilteredOptionalEffect: Effect that is a filtered value.
        - ZippedEffect: Effect that is a zipped value.
        - TappedEffect: Effect that is a tapped value.
        - RecoveredEffect: Effect that is a recovered value.
        - RetriedEffect: Effect that is a retried value.
        - MemorizedEffect: Effect that is a memorized value.
        - TimedOutEffect: Effect that is a timed out value.
        - SequenceEffect: Effect that is a sequence of effects.
        - ParallelEffect: Effect that is a parallel of effects.
        - RaceEffect: Effect that is a race of effects.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the effect.
        """

    def map(self, func: Callable[[T], U]) -> "Effect[U]":
        """Map the effect

        Args:
            func (Callable[[T], U]): The function to map the effect with.

        Returns:
            Effect[U]: The mapped effect wrapping the result with type U.
        """
        return MappedEffect(self, func)

    def flat_map(self, func: Callable[[T], "Effect[U]"]) -> "Effect[U]":
        """Flat map the effect

        Args:
            func (Callable[[T], Effect[U]]): The function to flat map the effect with.

        Returns:
            Effect[U]: The flat mapped effect wrapping the result with type U.
        """
        return FlatMappedEffect(self, func)

    def filter(self, predicate: Callable[[T], bool]) -> "Effect[T]":
        """Filter the effect

        Args:
            predicate (Callable[[T], bool]): The predicate to filter the effect with.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """
        return FilteredEffect(self, predicate)

    def find(self, predicate: Callable[[T], bool]) -> "Effect[T | None]":
        """Find the first element that satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to find the first element that satisfies.

        Returns:
            Effect[T | None]: The filtered optional effect wrapping the result with type T | None.
        """
        return FilteredOptionalEffect(self, predicate)

    def take_if(self, predicate: Callable[[T], bool]) -> "Effect[T]":
        """Take the value if it satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to take the value if it satisfies.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """
        return self.filter(predicate)

    def reject(self, predicate: Callable[[T], bool]) -> "Effect[T]":
        """Reject the value if it satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to reject the value if it satisfies.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """
        return FilteredEffect(self, lambda x: not predicate(x))

    def recover_from_none(self: "Effect[T | None]", default_factory: Callable[[], T]) -> "Effect[T]":
        """Convert Optional[T] -> T (default value if None)

        Args:
            default_factory (Callable[[], T]): The default factory to create the default value if the value is None.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """

        def recover() -> T:
            result = self.unwrap()
            return result if result is not None else default_factory()

        return FnEffect(recover)

    def or_else(self: "Effect[T | None]", alternative: "Effect[T]") -> "Effect[T]":
        """Execute the alternative effect if the value is None

        Args:
            alternative (Effect[T]): The alternative effect to execute if the value is None.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """

        def or_else_action() -> T:
            result = self.unwrap()
            return result if result is not None else alternative.unwrap()

        return FnEffect(or_else_action)

    def exists(self, predicate: Callable[[T], bool]) -> "Effect[bool]":
        """Check if the value satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to check if the value satisfies.

        Returns:
            Effect[bool]: The exists effect wrapping the result with type bool.
        """

        def exists_check() -> bool:
            try:
                result = self.unwrap()
                return predicate(result)
            except Exception:  # noqa: BLE001
                return False

        return FnEffect(exists_check)

    def zip_with(self, other: "Effect[U]", combiner: Callable[[T, U], Any]) -> "Effect[Any]":
        """Zip the effect with another effect

        Args:
            other (Effect[U]): The other effect to zip with.
            combiner (Callable[[T, U], Any]): The combiner function to zip the effects with.

        Returns:
            Effect[Any]: The zipped effect wrapping the result with type Any.
        """
        return ZippedEffect(self, other, combiner)

    def tap(self, side_effect: Callable[[T], None]) -> "Effect[T]":
        """Tap effect (side effect)

        Args:
            side_effect (Callable[[T], None]): The side effect to tap the effect with.

        Returns:
            Effect[T]: The tapped effect wrapping the result with type T.
        """
        return TappedEffect(self, side_effect)

    def recover(self, handler: Callable[[Exception], T]) -> "Effect[T]":
        """Recover the effect

        Args:
            handler (Callable[[Exception], T]): The handler function to recover the effect with.

        Returns:
            Effect[T]: The recovered effect wrapping the result with type T.
        """
        return RecoveredEffect(self, handler)

    def retry(self, max_retries: int = 3, delay: float = 0.1) -> "Effect[T]":
        """Retry the effect

        Args:
            max_retries (int): The maximum number of retries.
            delay (float): The delay between retries.

        Returns:
            Effect[T]: The retried effect wrapping the result with type T.
        """
        return RetriedEffect(self, max_retries, delay)

    def memorize(self) -> "Effect[T]":
        """Memorize the effect

        Returns:
            Effect[T]: The memorized effect wrapping the result with type T.
        """
        return MemorizedEffect(self)

    def timeout(self, seconds: float) -> "Effect[T]":
        """Timeout the effect

        Args:
            seconds (float): The number of seconds to timeout the effect.

        Returns:
            Effect[T]: The timed out effect wrapping the result with type T.
        """
        return TimedOutEffect(self, seconds)

    @classmethod
    def pure(cls, value: T) -> "Effect[T]":
        """Create a pure effect

        Args:
            value (T): The value to create the pure effect with.

        Returns:
            Effect[T]: The pure effect wrapping the result with type T.
        """
        return PureEffect(value)

    @classmethod
    def sequence(cls, effects: list["Effect[T]"]) -> "Effect[list[T]]":
        """Sequence the effects

        Args:
            effects (list[Effect[T]]): The effects to sequence.

        Returns:
            Effect[list[T]]: The sequence effect wrapping the result with type list[T].
        """
        return SequenceEffect(effects)

    @classmethod
    def parallel(cls, effects: list["Effect[T]"], max_workers: int | None = None) -> "Effect[list[T]]":
        """Parallel the effects

        Args:
            effects (list[Effect[T]]): The effects to parallel.
            max_workers (int | None): The maximum number of workers to use.

        Returns:
            Effect[list[T]]: The parallel effect wrapping the result with type list[T].
        """
        return ParallelEffect(effects, max_workers)

    @classmethod
    def race(cls, effects: list["Effect[T]"]) -> "Effect[T]":
        """Race the effects

        Args:
            effects (list[Effect[T]]): The effects to race.

        Returns:
            Effect[T]: The race effect wrapping the result with type T.
        """
        return RaceEffect(effects)


@dataclasses.dataclass(slots=True)
class FnEffect(Effect[T]):
    """Effect that is a function

    Args:
        func (Callable[[], T]): The function to execute the effect with.
    """

    func: Callable[[], T]

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the effect.
        """
        return self.func()


@dataclasses.dataclass(slots=True)
class PureEffect(Effect[T]):
    """Effect that is a pure value

    Args:
        value (T): The value to create the pure effect with.
    """

    value: T

    def unwrap(self) -> T:
        """Return the value

        Returns:
            T: The value of the effect.
        """
        return self.value


@dataclasses.dataclass(slots=True)
class MappedEffect(Generic[T, U], Effect[U]):
    """Effect that is a mapped value

    Args:
        source (Effect[T]): The source effect to map.
        mapper (Callable[[T], U]): The mapper function to map the source effect with.
    """

    source: Effect[T]
    mapper: Callable[[T], U]

    def unwrap(self) -> U:
        """Execute the effect and return the result

        Returns:
            U: The result of the mapped effect.
        """
        return self.mapper(self.source.unwrap())


@dataclasses.dataclass(slots=True)
class FlatMappedEffect(Generic[T, U], Effect[U]):
    """Effect that is a flat mapped value

    Args:
        source (Effect[T]): The source effect to flat map.
        mapper (Callable[[T], Effect[U]]): The mapper function to flat map the source effect with.
    """

    source: Effect[T]
    mapper: Callable[[T], Effect[U]]

    def unwrap(self) -> U:
        """Execute the effect and return the result

        Returns:
            U: The result of the flat mapped effect.
        """
        return self.mapper(self.source.unwrap()).unwrap()


@dataclasses.dataclass(slots=True)
class FilteredEffect(Generic[T], Effect[T]):
    """Effect that is a filtered value

    Args:
        source (Effect[T]): The source effect to filter.
        predicate (Callable[[T], bool]): The predicate function to filter the source effect with.
    """

    source: Effect[T]
    predicate: Callable[[T], bool]

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the filtered effect.
        """
        result = self.source.unwrap()
        if self.predicate(result):
            return result
        raise FilterError(result)


@dataclasses.dataclass(slots=True)
class FilteredOptionalEffect(Generic[T], Effect[T | None]):
    """Effect that is a filtered value

    Args:
        source (Effect[T]): The source effect to filter.
        predicate (Callable[[T], bool]): The predicate function to filter the source effect with.
    """

    source: Effect[T]
    predicate: Callable[[T], bool]

    def unwrap(self) -> T | None:
        """Execute the effect and return the result

        Returns:
            T | None: The result of the filtered optional effect.
        """
        result = self.source.unwrap()
        return result if self.predicate(result) else None


@dataclasses.dataclass(slots=True)
class ZippedEffect(Generic[T, U], Effect[Any]):
    """Effect that is a zipped value

    Args:
        left (Effect[T]): The left effect to zip.
        right (Effect[U]): The right effect to zip.
        combiner (Callable[[T, U], Any]): The combiner function to zip the effects with.
    """

    left: Effect[T]
    right: Effect[U]
    combiner: Callable[[T, U], Any]

    def unwrap(self) -> Any:  # noqa: ANN401
        """Execute the effect and return the result

        Returns:
            Any: The result of the zipped effect.
        """
        return self.combiner(self.left.unwrap(), self.right.unwrap())


@dataclasses.dataclass(slots=True)
class TappedEffect(Effect[T]):
    """Effect that is a tapped value

    Args:
        source (Effect[T]): The source effect to tap.
        side_effect (Callable[[T], None]): The side effect to tap the source effect with.
    """

    source: Effect[T]
    side_effect: Callable[[T], None]

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the tapped effect.
        """
        result = self.source.unwrap()
        self.side_effect(result)
        return result


@dataclasses.dataclass(slots=True)
class RecoveredEffect(Effect[T]):
    """Effect that is a recovered value

    Args:
        source (Effect[T]): The source effect to recover.
        handler (Callable[[Exception], T]): The handler function to recover the source effect with.
    """

    source: Effect[T]
    handler: Callable[[Exception], T]

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the recovered effect.
        """
        try:
            return self.source.unwrap()
        except Exception as e:  # noqa: BLE001
            return self.handler(e)


@dataclasses.dataclass(slots=True)
class RetriedEffect(Effect[T]):
    """Effect that is a retried value

    Args:
        source (Effect[T]): The source effect to retry.
        max_retries (int): The maximum number of retries.
        delay (float): The delay between retries.
    """

    source: Effect[T]
    max_retries: int
    delay: float

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the retried effect.
        """
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
    """Effect that is a memorized value

    Args:
        source (Effect[T]): The source effect to memorize.
    """

    source: Effect[T]
    _cached_result: T | None = dataclasses.field(default=None, init=False)
    _computed: bool = dataclasses.field(default=False, init=False)

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the memorized effect.
        """
        if not self._computed:
            self._cached_result = self.source.unwrap()
            self._computed = True
        return self._cached_result  # type: ignore[return-value]


@dataclasses.dataclass(slots=True)
class TimedOutEffect(Effect[T]):
    """Effect that is a timed out value

    Args:
        source (Effect[T]): The source effect to timeout.
        seconds (float): The number of seconds to timeout the effect.
    """

    source: Effect[T]
    seconds: float

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the timed out effect.
        """
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
    """Effect that is a sequence of effects

    Args:
        effects (list[Effect[T]]): The effects to sequence.
    """

    effects: list[Effect[T]]

    def unwrap(self) -> list[T]:
        """Execute the effect and return the result

        Returns:
            list[T]: The result of the sequence effect.
        """
        return [effect.unwrap() for effect in self.effects]


@dataclasses.dataclass(slots=True)
class ParallelEffect(Effect[list[T]]):
    """Effect that is a parallel of effects

    Args:
        effects (list[Effect[T]]): The effects to parallel.
        max_workers (int | None): The maximum number of workers to use.
    """

    effects: list[Effect[T]]
    max_workers: int | None = None

    def unwrap(self) -> list[T]:
        """Execute the effect and return the result

        Returns:
            list[T]: The result of the parallel effect.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: list[concurrent.futures.Future[T]] = [executor.submit(effect.unwrap) for effect in self.effects]
            return [future.result() for future in futures]


@dataclasses.dataclass(slots=True)
class RaceEffect(Effect[T]):
    """Effect that is a race of effects

    Args:
        effects (list[Effect[T]]): The effects to race.
    """

    effects: list[Effect[T]]

    def unwrap(self) -> T:
        """Execute the effect and return the result

        Returns:
            T: The result of the race effect.
        """
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
