"""Test the effect module."""

import time
import unittest

import pytest

from pyeffects.effect import (
    Effect,
    FilteredEffect,
    FilteredOptionalEffect,
    FlatMappedEffect,
    FnEffect,
    MappedEffect,
    MemorizedEffect,
    ParallelEffect,
    PureEffect,
    RaceEffect,
    RecoveredEffect,
    RetriedEffect,
    SequenceEffect,
    TappedEffect,
    TimedOutEffect,
    ZippedEffect,
)
from pyeffects.exception import FilterError


class TestPureEffect(unittest.TestCase):
    """Test PureEffect class."""

    def test_pure_effect_unwrap(self) -> None:
        """Test unwrap returns the wrapped value."""
        effect = PureEffect(42)
        result = effect.unwrap()
        assert result == 42

    def test_pure_effect_with_string(self) -> None:
        """Test with string value."""
        effect = PureEffect("hello")
        result = effect.unwrap()
        assert result == "hello"

    def test_pure_effect_with_none(self) -> None:
        """Test with None value."""
        effect = PureEffect(None)
        result = effect.unwrap()
        assert result is None


class TestFnEffect(unittest.TestCase):
    """Test FnEffect class."""

    def test_fn_effect_unwrap(self) -> None:
        """Test unwrap calls the function."""
        effect = FnEffect(lambda: 42)
        result = effect.unwrap()
        assert result == 42

    def test_fn_effect_with_side_effects(self) -> None:
        """Test function with side effects."""
        counter = 0

        def increment() -> int:
            nonlocal counter
            counter += 1
            return counter

        effect = FnEffect(increment)
        result1 = effect.unwrap()
        result2 = effect.unwrap()
        assert result1 == 1
        assert result2 == 2

    def test_fn_effect_with_exception(self) -> None:
        """Test function that raises exception."""
        msg = "Test error"

        def raise_error() -> int:
            raise ValueError(msg)

        effect = FnEffect(raise_error)
        with pytest.raises(ValueError, match=msg):
            effect.unwrap()


class TestMappedEffect(unittest.TestCase):
    """Test MappedEffect class."""

    def test_mapped_effect_basic(self) -> None:
        """Test basic mapping."""
        source = PureEffect(10)
        effect = MappedEffect(source, lambda x: x * 2)
        result = effect.unwrap()
        assert result == 20

    def test_mapped_effect_chaining(self) -> None:
        """Test chaining maps."""
        source = PureEffect(5)
        mapped1 = source.map(lambda x: x * 2)
        mapped2 = mapped1.map(lambda x: x + 1)
        result = mapped2.unwrap()
        assert result == 11

    def test_mapped_effect_with_exception(self) -> None:
        """Test mapping with exception in source."""
        msg = "Source error"

        def raise_error() -> int:
            raise ValueError(msg)

        source = FnEffect(raise_error)
        mapped = source.map(lambda x: x * 2)
        with pytest.raises(ValueError, match=msg):
            mapped.unwrap()


class TestFlatMappedEffect(unittest.TestCase):
    """Test FlatMappedEffect class."""

    def test_flat_mapped_effect_basic(self) -> None:
        """Test basic flat mapping."""
        source = PureEffect(10)
        effect = FlatMappedEffect(source, lambda x: PureEffect(x * 2))
        result = effect.unwrap()
        assert result == 20

    def test_flat_mapped_effect_chaining(self) -> None:
        """Test chaining flat maps."""
        source = PureEffect(5)
        flat_mapped = source.flat_map(lambda x: PureEffect(x * 2))
        result = flat_mapped.unwrap()
        assert result == 10

    def test_flat_mapped_effect_nested(self) -> None:
        """Test nested effects."""
        source = PureEffect(3)
        nested = source.flat_map(lambda x: PureEffect(x).map(lambda y: y * 2))
        result = nested.unwrap()
        assert result == 6


class TestFilteredEffect(unittest.TestCase):
    """Test FilteredEffect class."""

    def test_filtered_effect_pass(self) -> None:
        """Test filter that passes."""
        source = PureEffect(10)
        effect = FilteredEffect(source, lambda x: x > 5)
        result = effect.unwrap()
        assert result == 10

    def test_filtered_effect_fail(self) -> None:
        """Test filter that fails."""
        source = PureEffect(3)
        effect = FilteredEffect(source, lambda x: x > 5)
        with pytest.raises(FilterError) as cm:
            effect.unwrap()
        assert cm.value.value == 3

    def test_filter_method(self) -> None:
        """Test filter method on Effect."""
        source = PureEffect(10)
        filtered = source.filter(lambda x: x > 5)
        result = filtered.unwrap()
        assert result == 10

    def test_take_if_method(self) -> None:
        """Test take_if method."""
        source = PureEffect(10)
        taken = source.take_if(lambda x: x > 5)
        result = taken.unwrap()
        assert result == 10

    def test_reject_method(self) -> None:
        """Test reject method."""
        source = PureEffect(10)
        rejected = source.reject(lambda x: x < 5)
        result = rejected.unwrap()
        assert result == 10


class TestFilteredOptionalEffect(unittest.TestCase):
    """Test FilteredOptionalEffect class."""

    def test_filtered_optional_effect_pass(self) -> None:
        """Test filter that passes."""
        source = PureEffect(10)
        effect = FilteredOptionalEffect(source, lambda x: x > 5)
        result = effect.unwrap()
        assert result == 10

    def test_filtered_optional_effect_fail(self) -> None:
        """Test filter that fails."""
        source = PureEffect(3)
        effect = FilteredOptionalEffect(source, lambda x: x > 5)
        result = effect.unwrap()
        assert result is None

    def test_find_method(self) -> None:
        """Test find method."""
        source = PureEffect(10)
        found = source.find(lambda x: x > 5)
        result = found.unwrap()
        assert result == 10

    def test_find_method_not_found(self) -> None:
        """Test find method when not found."""
        source = PureEffect(3)
        found = source.find(lambda x: x > 5)
        result = found.unwrap()
        assert result is None


class TestZippedEffect(unittest.TestCase):
    """Test ZippedEffect class."""

    def test_zipped_effect_basic(self) -> None:
        """Test basic zipping."""
        left = PureEffect(10)
        right = PureEffect(20)
        effect = ZippedEffect(left, right, lambda x, y: x + y)
        result = effect.unwrap()
        assert result == 30

    def test_zip_with_method(self) -> None:
        """Test zip_with method."""
        left = PureEffect(5)
        right = PureEffect(3)
        zipped = left.zip_with(right, lambda x, y: x * y)
        result = zipped.unwrap()
        assert result == 15

    def test_zipped_effect_with_different_types(self) -> None:
        """Test zipping with different types."""
        left = PureEffect(10)
        right = PureEffect("hello")
        zipped = left.zip_with(right, lambda x, y: f"{y}_{x}")
        result = zipped.unwrap()
        assert result == "hello_10"


class TestTappedEffect(unittest.TestCase):
    """Test TappedEffect class."""

    def test_tapped_effect_basic(self) -> None:
        """Test basic tapping."""
        side_effects: list[int] = []

        def side_effect(x: int) -> None:
            side_effects.append(x)

        source = PureEffect(42)
        effect = TappedEffect(source, side_effect)
        result = effect.unwrap()

        assert result == 42
        assert side_effects == [42]

    def test_tap_method(self) -> None:
        """Test tap method."""
        side_effects: list[int] = []

        def side_effect(x: int) -> None:
            side_effects.append(x * 2)

        source = PureEffect(10)
        tapped = source.tap(side_effect)
        result = tapped.unwrap()

        assert result == 10
        assert side_effects == [20]

    def test_tapped_effect_preserves_original(self) -> None:
        """Test tapped effect preserves original value."""

        def side_effect(x: int) -> None:
            pass

        source = PureEffect(100)
        tapped = source.tap(side_effect)
        result = tapped.unwrap()
        assert result == 100


class TestRecoveredEffect(unittest.TestCase):
    """Test RecoveredEffect class."""

    def test_recovered_effect_no_exception(self) -> None:
        """Test recovered effect when no exception occurs."""
        source = PureEffect(42)
        effect = RecoveredEffect(source, lambda _: 0)
        result = effect.unwrap()
        assert result == 42

    def test_recovered_effect_with_exception(self) -> None:
        """Test recovered effect when exception occurs."""
        msg = "Test error"

        def raise_error() -> int:
            raise ValueError(msg)

        source = FnEffect(raise_error)
        effect = RecoveredEffect(source, lambda _: 999)
        result = effect.unwrap()
        assert result == 999

    def test_recover_method(self) -> None:
        """Test recover method."""
        msg = "Test error"

        def raise_error() -> int:
            raise ValueError(msg)

        source = FnEffect(raise_error)
        recovered = source.recover(lambda _: 123)
        result = recovered.unwrap()
        assert result == 123

    def test_recovered_effect_handler_access_exception(self) -> None:
        """Test handler can access the exception."""
        msg = "Test error"

        def raise_error() -> int:
            raise ValueError(msg)

        source = FnEffect(raise_error)
        recovered = source.recover(lambda _: -1)
        result = recovered.unwrap()
        assert result == -1


class TestRetriedEffect(unittest.TestCase):
    """Test RetriedEffect class."""

    def test_retried_effect_success_first_try(self) -> None:
        """Test success on first try."""
        source = PureEffect(42)
        effect = RetriedEffect(source, max_retries=3, delay=0.1)
        result = effect.unwrap()
        assert result == 42

    def test_retried_effect_success_after_retries(self) -> None:
        """Test success after some retries."""
        call_count = 0
        msg = "Temporary error"

        def flaky_function() -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(msg)
            return 42

        source = FnEffect(flaky_function)
        effect = RetriedEffect(source, max_retries=5, delay=0.01)
        result = effect.unwrap()
        assert result == 42
        assert call_count == 3

    def test_retried_effect_max_retries_exceeded(self) -> None:
        """Test when max retries is exceeded."""
        msg = "Always fails"

        def always_fail() -> int:
            raise ValueError(msg)

        source = FnEffect(always_fail)
        effect = RetriedEffect(source, max_retries=2, delay=0.01)
        with pytest.raises(ValueError, match=msg):
            effect.unwrap()

    def test_retry_method(self) -> None:
        """Test retry method."""
        call_count = 0
        msg = "Temporary error"

        def flaky_function() -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError(msg)
            return 100

        source = FnEffect(flaky_function)
        retried = source.retry(max_retries=3, delay=0.01)
        result = retried.unwrap()
        assert result == 100


class TestMemorizedEffect(unittest.TestCase):
    """Test MemorizedEffect class."""

    def test_memorized_effect_caches_result(self) -> None:
        """Test that result is cached."""
        call_count = 0

        def expensive_function() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        source = FnEffect(expensive_function)
        effect = MemorizedEffect(source)

        result1 = effect.unwrap()
        result2 = effect.unwrap()

        assert result1 == 42
        assert result2 == 42
        assert call_count == 1

    def test_memorize_method(self) -> None:
        """Test memorize method."""
        call_count = 0

        def expensive_function() -> int:
            nonlocal call_count
            call_count += 1
            return 100

        source = FnEffect(expensive_function)
        memorized = source.memorize()

        result1 = memorized.unwrap()
        result2 = memorized.unwrap()

        assert result1 == 100
        assert result2 == 100
        assert call_count == 1

    def test_memorized_effect_with_exception(self) -> None:
        """Test memorized effect with exception."""
        call_count = 0
        msg = "Test error"

        def failing_function() -> int:
            nonlocal call_count
            call_count += 1
            raise ValueError(msg)

        source = FnEffect(failing_function)
        memorized = source.memorize()

        with pytest.raises(ValueError, match=msg):
            memorized.unwrap()

        with pytest.raises(ValueError, match=msg):
            memorized.unwrap()

        assert call_count == 2


class TestTimedOutEffect(unittest.TestCase):
    """Test TimedOutEffect class."""

    def test_timed_out_effect_within_timeout(self) -> None:
        """Test effect completes within timeout."""
        source = PureEffect(42)
        effect = TimedOutEffect(source, seconds=1.0)
        result = effect.unwrap()
        assert result == 42

    def test_timed_out_effect_exceeds_timeout(self) -> None:
        """Test effect exceeds timeout."""

        def slow_function() -> int:
            time.sleep(0.2)
            return 42

        source = FnEffect(slow_function)
        effect = TimedOutEffect(source, seconds=0.1)
        with pytest.raises(TimeoutError):
            effect.unwrap()

    def test_timeout_method(self) -> None:
        """Test timeout method."""
        source = PureEffect(100)
        timed_out = source.timeout(seconds=1.0)
        result = timed_out.unwrap()
        assert result == 100


class TestSequenceEffect(unittest.TestCase):
    """Test SequenceEffect class."""

    def test_sequence_effect_basic(self) -> None:
        """Test basic sequence."""
        effects: list[Effect[int]] = [
            PureEffect(1),
            PureEffect(2),
            PureEffect(3),
        ]
        effect = SequenceEffect(effects)
        result = effect.unwrap()
        assert result == [1, 2, 3]

    def test_sequence_class_method(self) -> None:
        """Test sequence class method."""
        effects: list[Effect[int]] = [
            PureEffect(10),
            PureEffect(20),
            PureEffect(30),
        ]
        sequenced: Effect[list[int]] = Effect[int].sequence(effects)
        result = sequenced.unwrap()
        assert result == [10, 20, 30]

    def test_sequence_effect_empty_list(self) -> None:
        """Test sequence with empty list."""
        effects: list[Effect[int]] = []
        effect = SequenceEffect(effects)
        result = effect.unwrap()
        assert result == []

    def test_sequence_effect_with_exception(self) -> None:
        """Test sequence with exception in one effect."""
        msg = "Test error"

        def raise_error() -> int:
            raise ValueError(msg)

        effects: list[Effect[int]] = [
            PureEffect(1),
            FnEffect(raise_error),
            PureEffect(3),
        ]
        effect = SequenceEffect(effects)
        with pytest.raises(ValueError, match=msg):
            effect.unwrap()


class TestParallelEffect(unittest.TestCase):
    """Test ParallelEffect class."""

    def test_parallel_effect_basic(self) -> None:
        """Test basic parallel execution."""
        effects: list[Effect[int]] = [
            PureEffect(1),
            PureEffect(2),
            PureEffect(3),
        ]
        effect = ParallelEffect(effects)
        result = effect.unwrap()
        assert result == [1, 2, 3]

    def test_parallel_class_method(self) -> None:
        """Test parallel class method."""
        effects: list[Effect[int]] = [
            PureEffect(10),
            PureEffect(20),
            PureEffect(30),
        ]
        parallel: Effect[list[int]] = Effect[int].parallel(effects)
        result = parallel.unwrap()
        assert result == [10, 20, 30]

    def test_parallel_effect_with_max_workers(self) -> None:
        """Test parallel with max_workers."""
        effects: list[Effect[int]] = [
            PureEffect(1),
            PureEffect(2),
        ]
        effect = ParallelEffect(effects, max_workers=1)
        result = effect.unwrap()
        assert result == [1, 2]

    def test_parallel_effect_empty_list(self) -> None:
        """Test parallel with empty list."""
        effects: list[Effect[int]] = []
        effect = ParallelEffect(effects)
        result = effect.unwrap()
        assert result == []


class TestRaceEffect(unittest.TestCase):
    """Test RaceEffect class."""

    def test_race_effect_basic(self) -> None:
        """Test basic race."""
        effects: list[Effect[int]] = [
            PureEffect(1),
            PureEffect(2),
            PureEffect(3),
        ]
        effect = RaceEffect(effects)
        result = effect.unwrap()
        assert result in [1, 2, 3]

    def test_race_class_method(self) -> None:
        """Test race class method."""
        effects: list[Effect[int]] = [
            PureEffect(10),
            PureEffect(20),
        ]
        race: Effect[int] = Effect[int].race(effects)
        result = race.unwrap()
        assert result in [10, 20]

    def test_race_effect_with_delays(self) -> None:
        """Test race with different delays."""

        def fast_function() -> int:
            return 1

        def slow_function() -> int:
            time.sleep(0.1)
            return 2

        effects: list[Effect[int]] = [
            FnEffect(slow_function),
            FnEffect(fast_function),
        ]
        effect = RaceEffect(effects)
        result = effect.unwrap()
        assert result == 1


class TestEffectOptionalMethods(unittest.TestCase):
    """Test optional-related methods on Effect."""

    def test_recover_from_none_with_none(self) -> None:
        """Test recover_from_none with None value."""

        def default_factory() -> int:
            return 42

        source = PureEffect[int | None](None)
        recovered = source.recover_from_none(default_factory)
        result = recovered.unwrap()
        assert result == 42

    def test_recover_from_none_with_value(self) -> None:
        """Test recover_from_none with non-None value."""

        def default_factory() -> int:
            return 42

        source = PureEffect[int | None](10)
        recovered = source.recover_from_none(default_factory)
        result = recovered.unwrap()
        assert result == 10

    def test_or_else_with_none(self) -> None:
        """Test or_else with None value."""
        source = PureEffect[int | None](None)
        alternative = PureEffect[int | None](99)
        result_effect: Effect[int | None] = source.or_else(alternative)
        result = result_effect.unwrap()
        assert result == 99

    def test_or_else_with_value(self) -> None:
        """Test or_else with non-None value."""
        source = PureEffect[int | None](10)
        alternative = PureEffect[int | None](99)
        result_effect: Effect[int | None] = source.or_else(alternative)
        result = result_effect.unwrap()
        assert result == 10

    def test_exists_true(self) -> None:
        """Test exists method returns True."""
        source = PureEffect(10)
        exists = source.exists(lambda x: x > 5)
        result = exists.unwrap()
        assert result

    def test_exists_false(self) -> None:
        """Test exists method returns False."""
        source = PureEffect(3)
        exists = source.exists(lambda x: x > 5)
        result = exists.unwrap()
        assert not result

    def test_exists_with_exception(self) -> None:
        """Test exists method with exception."""
        msg = "Test error"

        def raise_error() -> int:
            raise ValueError(msg)

        source = FnEffect(raise_error)
        exists = source.exists(lambda x: x > 5)
        result = exists.unwrap()
        assert not result


class TestEffectPureMethod(unittest.TestCase):
    """Test Effect.pure class method."""

    def test_pure_class_method(self) -> None:
        """Test pure class method."""
        effect: Effect[int] = Effect[int].pure(value=42)
        result = effect.unwrap()
        assert result == 42

    def test_pure_class_method_with_string(self) -> None:
        """Test pure class method with string."""
        effect: Effect[str] = Effect[str].pure(value="hello")
        result = effect.unwrap()
        assert result == "hello"


if __name__ == "__main__":
    unittest.main()
