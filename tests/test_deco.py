"""Tests for decorator functions."""

import unittest

import pytest
from pyresults import Err, Ok, Result

from pyeffects.base import Effect
from pyeffects.deco import effect, io, safe


class TestEffectDecorator(unittest.TestCase):
    """Test effect decorator function."""

    def test_effect_decorator_basic(self) -> None:
        """Test basic effect decorator."""

        @effect
        def add_numbers(x: int, y: int) -> int:
            return x + y

        result_effect = add_numbers(3, 5)
        assert isinstance(result_effect, Effect)
        result = result_effect.unwrap()
        assert result == 8

    def test_effect_decorator_no_args(self) -> None:
        """Test effect decorator with no arguments."""

        @effect
        def get_constant() -> int:
            return 42

        result_effect = get_constant()
        assert isinstance(result_effect, Effect)
        result = result_effect.unwrap()
        assert result == 42

    def test_effect_decorator_with_side_effects(self) -> None:
        """Test effect decorator with side effects."""
        counter = 0

        @effect
        def increment_counter() -> int:
            nonlocal counter
            counter += 1
            return counter

        effect1 = increment_counter()
        effect2 = increment_counter()

        result1 = effect1.unwrap()
        result2 = effect2.unwrap()

        assert result1 == 1
        assert result2 == 2

    def test_effect_decorator_with_exception(self) -> None:
        """Test effect decorator with exception."""
        msg = "Test error"

        @effect
        def raise_error() -> int:
            raise ValueError(msg)

        result_effect = raise_error()
        with pytest.raises(ValueError, match=msg):
            result_effect.unwrap()

    def test_effect_decorator_preserves_function_name(self) -> None:
        """Test that decorator preserves function name."""

        @effect
        def original_function() -> int:
            return 42

        assert original_function.__name__ == "original_function"

    def test_effect_decorator_with_kwargs(self) -> None:
        """Test effect decorator with keyword arguments."""

        @effect
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result_effect = greet("Alice", greeting="Hi")
        result = result_effect.unwrap()
        assert result == "Hi, Alice!"

    def test_effect_decorator_with_mixed_args(self) -> None:
        """Test effect decorator with mixed positional and keyword arguments."""

        @effect
        def complex_function(x: int, y: int, multiplier: int = 1) -> int:
            return (x + y) * multiplier

        result_effect = complex_function(3, 4, multiplier=2)
        result = result_effect.unwrap()
        assert result == 14

    def test_effect_decorator_lazy_evaluation(self) -> None:
        """Test that effect decorator provides lazy evaluation."""
        call_count = 0

        @effect
        def count_calls() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result_effect = count_calls()
        assert call_count == 0  # Should not be called yet

        result = result_effect.unwrap()
        assert call_count == 1
        assert result == 1


class TestIoDecorator(unittest.TestCase):
    """Test io decorator function."""

    def test_io_decorator_basic(self) -> None:
        """Test basic io decorator."""

        @io
        def read_file(filename: str) -> str:
            return f"Contents of {filename}"

        result_effect = read_file("test.txt")
        assert isinstance(result_effect, Effect)
        result = result_effect.unwrap()
        assert result == "Contents of test.txt"

    def test_io_decorator_is_alias_for_effect(self) -> None:
        """Test that io decorator is an alias for effect decorator."""

        @io
        def io_function() -> int:
            return 42

        @effect
        def effect_function() -> int:
            return 42

        io_result = io_function()
        effect_result = effect_function()

        assert isinstance(io_result, Effect)
        assert isinstance(effect_result, Effect)
        assert io_result.unwrap() == effect_result.unwrap()

    def test_io_decorator_with_side_effects(self) -> None:
        """Test io decorator with side effects."""
        outputs: list[str] = []

        @io
        def write_to_output(message: str) -> str:
            outputs.append(message)
            return f"Wrote: {message}"

        effect = write_to_output("Hello")
        result = effect.unwrap()

        assert result == "Wrote: Hello"
        assert outputs == ["Hello"]

    def test_io_decorator_preserves_function_name(self) -> None:
        """Test that io decorator preserves function name."""

        @io
        def io_operation() -> str:
            return "IO result"

        assert io_operation.__name__ == "io_operation"

    def test_io_decorator_lazy_evaluation(self) -> None:
        """Test that io decorator provides lazy evaluation."""
        call_count = 0

        @io
        def count_io_calls() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        effect = count_io_calls()
        assert call_count == 0  # Should not be called yet

        result = effect.unwrap()
        assert call_count == 1
        assert result == 1


class TestSafeDecorator(unittest.TestCase):
    """Test safe decorator function."""

    def test_safe_decorator_basic(self) -> None:
        """Test basic safe decorator."""

        @safe
        def safe_operation(x: int) -> Result[int, str]:
            if x > 0:
                return Ok(x * 2)
            return Err("Negative input")

        result_effect = safe_operation(5)
        assert isinstance(result_effect, Effect)
        result = result_effect.unwrap()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() == 10

    def test_safe_decorator_with_error_result(self) -> None:
        """Test safe decorator with error result."""

        @safe
        def safe_operation(x: int) -> Result[int, str]:
            if x > 0:
                return Ok(x * 2)
            return Err("Negative input")

        result_effect = safe_operation(-1)
        result = result_effect.unwrap()
        assert isinstance(result, Result)
        assert result.is_err()
        assert result.unwrap_err() == "Negative input"

    def test_safe_decorator_with_exception(self) -> None:
        """Test safe decorator when function raises exception."""
        msg = "Unexpected error"

        @safe
        def failing_safe_operation() -> Result[int, str]:
            raise ValueError(msg)

        effect = failing_safe_operation()
        with pytest.raises(ValueError, match=msg):
            effect.unwrap()

    def test_safe_decorator_preserves_function_name(self) -> None:
        """Test that safe decorator preserves function name."""

        @safe
        def safe_function() -> Result[int, str]:
            return Ok(42)

        assert safe_function.__name__ == "safe_function"

    def test_safe_decorator_with_arguments(self) -> None:
        """Test safe decorator with arguments."""

        @safe
        def divide_safely(x: int, y: int) -> Result[float, str]:
            if y == 0:
                return Err("Division by zero")
            return Ok(x / y)

        result_effect = divide_safely(10, 2)
        result = result_effect.unwrap()
        assert result.is_ok()
        assert result.unwrap() == 5.0

    def test_safe_decorator_with_error_case(self) -> None:
        """Test safe decorator with error case."""

        @safe
        def divide_safely(x: int, y: int) -> Result[float, str]:
            if y == 0:
                return Err("Division by zero")
            return Ok(x / y)

        result_effect = divide_safely(10, 0)
        result = result_effect.unwrap()
        assert result.is_err()
        assert result.unwrap_err() == "Division by zero"

    def test_safe_decorator_lazy_evaluation(self) -> None:
        """Test that safe decorator provides lazy evaluation."""
        call_count = 0

        @safe
        def count_safe_calls() -> Result[int, str]:
            nonlocal call_count
            call_count += 1
            return Ok(call_count)

        effect = count_safe_calls()
        assert call_count == 0  # Should not be called yet
        result = effect.unwrap()
        assert result.is_ok()
        assert result.unwrap() == 1

    def test_safe_decorator_with_kwargs(self) -> None:
        """Test safe decorator with keyword arguments."""

        @safe
        def create_user(name: str, age: int = 18) -> Result[dict[str, str | int], str]:
            if age < 0:
                return Err("Invalid age")
            return Ok({"name": name, "age": age})

        result_effect = create_user("Alice", age=25)
        result = result_effect.unwrap()
        assert result.is_ok()
        assert result.unwrap() == {"name": "Alice", "age": 25}

    def test_safe_decorator_returns_effect_of_result(self) -> None:
        """Test that safe decorator returns Effect[Result[T, E]]."""

        @safe
        def simple_operation() -> Result[str, int]:
            return Ok("success")

        effect = simple_operation()
        assert isinstance(effect, Effect)

        result = effect.unwrap()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() == "success"


class TestDecoratorsIntegration(unittest.TestCase):
    """Test decorators integration and combinations."""

    def test_chaining_effects_from_decorators(self) -> None:
        """Test chaining effects created by decorators."""

        @effect
        def add_one(x: int) -> int:
            return x + 1

        @effect
        def multiply_by_two(x: int) -> int:
            return x * 2

        initial_effect = add_one(5)
        chained_effect = initial_effect.flat_map(lambda x: multiply_by_two(x))
        result = chained_effect.unwrap()
        assert result == 12  # (5 + 1) * 2

    def test_mapping_effects_from_decorators(self) -> None:
        """Test mapping effects created by decorators."""

        @effect
        def get_name() -> str:
            return "Alice"

        name_effect = get_name()
        greeting_effect = name_effect.map(lambda name: f"Hello, {name}!")
        result = greeting_effect.unwrap()
        assert result == "Hello, Alice!"

    def test_combining_different_decorators(self) -> None:
        """Test combining effects from different decorators."""

        @effect
        def get_value() -> int:
            return 10

        @io
        def process_value(x: int) -> str:
            return f"Processed: {x}"

        value_effect = get_value()
        processed_effect = value_effect.flat_map(lambda x: process_value(x))
        result = processed_effect.unwrap()
        assert result == "Processed: 10"


if __name__ == "__main__":
    unittest.main()
