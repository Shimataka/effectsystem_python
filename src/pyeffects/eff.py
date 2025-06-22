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

from pyresults import Err, Ok, Result

from pyeffects.exception import FilterError

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")
P = ParamSpec("P")


class Eff(Generic[T, E], ABC):
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

    def run(self) -> Result[T, E]:
        """Execute the effect and return the result wrapped in Result

        Returns:
            Result[T, E]: The result of the effect wrapped in Result.
        """
        try:
            return Ok(self.unwrap())
        except Exception as e:  # noqa: BLE001
            return Err(e)  # type: ignore[arg-type]

    def map(self, func: Callable[[T], U]) -> "Eff[U, E]":
        """Map the effect

        Args:
            func (Callable[[T], U]): The function to map the effect with.

        Returns:
            Effect[U]: The mapped effect wrapping the result with type U.
        """
        return MappedEff(self, func)

    def flat_map(self, func: Callable[[T], "Eff[U, E]"]) -> "Eff[U, E]":
        """Flat map the effect

        Args:
            func (Callable[[T], Effect[U]]): The function to flat map the effect with.

        Returns:
            Effect[U]: The flat mapped effect wrapping the result with type U.
        """
        return FlatMappedEff(self, func)

    def filter(self, predicate: Callable[[T], bool]) -> "Eff[T, E]":
        """Filter the effect

        Args:
            predicate (Callable[[T], bool]): The predicate to filter the effect with.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """
        return FilteredEff(self, predicate)

    def find(self, predicate: Callable[[T], bool]) -> "Eff[T | None, E]":
        """Find the first element that satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to find the first element that satisfies.

        Returns:
            Effect[T | None]: The filtered optional effect wrapping the result with type T | None.
        """
        return FilteredOptionalEff(self, predicate)

    def take_if(self, predicate: Callable[[T], bool]) -> "Eff[T, E]":
        """Take the value if it satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to take the value if it satisfies.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """
        return self.filter(predicate)

    def reject(self, predicate: Callable[[T], bool]) -> "Eff[T, E]":
        """Reject the value if it satisfies the predicate

        Args:
            predicate (Callable[[T], bool]): The predicate to reject the value if it satisfies.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """
        return FilteredEff(self, lambda x: not predicate(x))

    def recover_from_error(self: "Eff[T, E]", default_factory: Callable[[], T]) -> "Eff[T, E]":
        """Convert Err[T] -> T (default value if error)

        Args:
            default_factory (Callable[[], T]): The default factory to create the default value if the value is an error

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """

        def recover() -> T:
            try:
                return self.unwrap()
            except Exception:  # noqa: BLE001
                return default_factory()

        return FnEff(recover)

    def or_else(self, alternative: "Eff[T, E]") -> "Eff[T, E]":
        """Execute the alternative effect if the value is None

        Args:
            alternative (Effect[T]): The alternative effect to execute if the value is None.

        Returns:
            Effect[T]: The filtered effect wrapping the result with type T.
        """

        def or_else_action() -> T:
            try:
                result = self.unwrap()
                if result is not None:
                    return result
                return alternative.unwrap()
            except Exception:  # noqa: BLE001
                return alternative.unwrap()

        return FnEff(or_else_action)

    def exists(self, predicate: Callable[[T], bool]) -> "Eff[bool, E]":
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

        return FnEff(exists_check)

    def zip_with(self, other: "Eff[U, E]", combiner: Callable[[T, U], Any]) -> "Eff[Any, E]":
        """Zip the effect with another effect

        Args:
            other (Effect[U]): The other effect to zip with.
            combiner (Callable[[T, U], Any]): The combiner function to zip the effects with.

        Returns:
            Effect[Any]: The zipped effect wrapping the result with type Any.
        """
        return ZippedEff(self, other, combiner)

    def tap(self, side_effect: Callable[[T], None]) -> "Eff[T, E]":
        """Tap effect (side effect)

        Args:
            side_effect (Callable[[T], None]): The side effect to tap the effect with.

        Returns:
            Effect[T]: The tapped effect wrapping the result with type T.
        """
        return TappedEff(self, side_effect)

    def recover(self, handler: Callable[[Exception], T]) -> "Eff[T, E]":
        """Recover the effect

        Args:
            handler (Callable[[Exception], T]): The handler function to recover the effect with.

        Returns:
            Effect[T]: The recovered effect wrapping the result with type T.
        """
        return RecoveredEff(self, handler)

    def retry(self, max_retries: int = 3, delay: float = 0.1) -> "Eff[T, E]":
        """Retry the effect

        Args:
            max_retries (int): The maximum number of retries.
            delay (float): The delay between retries.

        Returns:
            Effect[T]: The retried effect wrapping the result with type T.
        """
        return RetriedEff(self, max_retries, delay)

    def memorize(self) -> "Eff[T, E]":
        """Memorize the effect

        Returns:
            Effect[T]: The memorized effect wrapping the result with type T.
        """
        return MemorizedEff(self)

    def timeout(self, seconds: float) -> "Eff[T, E]":
        """Timeout the effect

        Args:
            seconds (float): The number of seconds to timeout the effect.

        Returns:
            Effect[T]: The timed out effect wrapping the result with type T.
        """
        return TimedOutEff(self, seconds)

    @classmethod
    def pure(cls, value: T) -> "Eff[T, E]":
        """Create a pure effect

        Args:
            value (T): The value to create the pure effect with.

        Returns:
            Effect[T]: The pure effect wrapping the result with type T.
        """
        return PureEff(value)

    @classmethod
    def sequence(cls, effects: list["Eff[T, E]"]) -> "Eff[list[T], E]":
        """Sequence the effects

        Args:
            effects (list[Effect[T]]): The effects to sequence.

        Returns:
            Effect[list[T]]: The sequence effect wrapping the result with type list[T].
        """
        return SequenceEff(effects)

    @classmethod
    def parallel(cls, effects: list["Eff[T, E]"], max_workers: int | None = None) -> "Eff[list[T], E]":
        """Parallel the effects

        Args:
            effects (list[Effect[T]]): The effects to parallel.
            max_workers (int | None): The maximum number of workers to use.

        Returns:
            Effect[list[T]]: The parallel effect wrapping the result with type list[T].
        """
        return ParallelEff(effects, max_workers)

    @classmethod
    def race(cls, effects: list["Eff[T, E]"]) -> "Eff[T, E]":
        """Race the effects

        Args:
            effects (list[Effect[T]]): The effects to race.

        Returns:
            Effect[T]: The race effect wrapping the result with type T.
        """
        return RaceEff(effects)


@dataclasses.dataclass(slots=True)
class FnEff(Eff[T, E]):
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
class PureEff(Eff[T, E]):
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
class MappedEff(Generic[T, U, E], Eff[U, E]):
    """Effect that is a mapped value

    Args:
        source (Effect[T]): The source effect to map.
        mapper (Callable[[T], U]): The mapper function to map the source effect with.
    """

    source: Eff[T, E]
    mapper: Callable[[T], U]

    def unwrap(self) -> U:
        """Execute the effect and return the result

        Returns:
            U: The result of the mapped effect.
        """
        result = self.source.unwrap()
        return self.mapper(result)


@dataclasses.dataclass(slots=True)
class FlatMappedEff(Generic[T, U, E], Eff[U, E]):
    """Effect that is a flat mapped value

    Args:
        source (Effect[T]): The source effect to flat map.
        mapper (Callable[[T], Effect[U]]): The mapper function to flat map the source effect with.
    """

    source: Eff[T, E]
    mapper: Callable[[T], Eff[U, E]]

    def unwrap(self) -> U:
        """Execute the effect and return the result

        Returns:
            U: The result of the flat mapped effect.
        """
        result = self.source.unwrap()
        return self.mapper(result).unwrap()


@dataclasses.dataclass(slots=True)
class FilteredEff(Generic[T, E], Eff[T, E]):
    """Effect that is a filtered value

    Args:
        source (Effect[T]): The source effect to filter.
        predicate (Callable[[T], bool]): The predicate function to filter the source effect with.
    """

    source: Eff[T, E]
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
class FilteredOptionalEff(Generic[T, E], Eff[T | None, E]):
    """Effect that is a filtered value

    Args:
        source (Effect[T]): The source effect to filter.
        predicate (Callable[[T], bool]): The predicate function to filter the source effect with.
    """

    source: Eff[T, E]
    predicate: Callable[[T], bool]

    def unwrap(self) -> T | None:
        """Execute the effect and return the result

        Returns:
            T | None: The result of the filtered optional effect.
        """
        result = self.source.unwrap()
        return result if self.predicate(result) else None


@dataclasses.dataclass(slots=True)
class ZippedEff(Generic[T, U, E, F], Eff[Any, E | F]):
    """Effect that is a zipped value

    Args:
        left (Effect[T, E]): The left effect to zip.
        right (Effect[U, F]): The right effect to zip.
        combiner (Callable[[T, U], Any]): The combiner function to zip the effects with.
    """

    left: Eff[T, E]
    right: Eff[U, F]
    combiner: Callable[[T, U], Any]

    def unwrap(self) -> Any:  # noqa: ANN401
        """Execute the effect and return the result

        Returns:
            Any: The result of the zipped effect.
        """
        return self.combiner(self.left.unwrap(), self.right.unwrap())


@dataclasses.dataclass(slots=True)
class TappedEff(Eff[T, E]):
    """Effect that is a tapped value

    Args:
        source (Effect[T]): The source effect to tap.
        side_effect (Callable[[T], None]): The side effect to tap the source effect with.
    """

    source: Eff[T, E]
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
class RecoveredEff(Eff[T, E]):
    """Effect that is a recovered value

    Args:
        source (Effect[T]): The source effect to recover.
        handler (Callable[[Exception], T]): The handler function to recover the source effect with.
    """

    source: Eff[T, E]
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
class RetriedEff(Eff[T, E]):
    """Effect that is a retried value

    Args:
        source (Effect[T]): The source effect to retry.
        max_retries (int): The maximum number of retries.
        delay (float): The delay between retries.
    """

    source: Eff[T, E]
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
class MemorizedEff(Eff[T, E]):
    """Effect that is a memorized value

    Args:
        source (Effect[T]): The source effect to memorize.
    """

    source: Eff[T, E]
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
class TimedOutEff(Eff[T, E]):
    """Effect that is a timed out value

    Args:
        source (Effect[T]): The source effect to timeout.
        seconds (float): The number of seconds to timeout the effect.
    """

    source: Eff[T, E]
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
class SequenceEff(Eff[list[T], E]):
    """Effect that is a sequence of effects

    Args:
        effects (list[Effect[T]]): The effects to sequence.
    """

    effects: list[Eff[T, E]]

    def unwrap(self) -> list[T]:
        """Execute the effect and return the result

        Returns:
            list[T]: The result of the sequence effect.
        """
        return [effect.unwrap() for effect in self.effects]


@dataclasses.dataclass(slots=True)
class ParallelEff(Eff[list[T], E]):
    """Effect that is a parallel of effects

    Args:
        effects (list[Effect[T]]): The effects to parallel.
        max_workers (int | None): The maximum number of workers to use.
    """

    effects: list[Eff[T, E]]
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
class RaceEff(Eff[T, E]):
    """Effect that is a race of effects

    Args:
        effects (list[Effect[T]]): The effects to race.
    """

    effects: list[Eff[T, E]]

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
