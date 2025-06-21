"""Tests for custom exceptions."""

import unittest

import pytest

from pyeffects.exception import FilterError


class TestFilterError(unittest.TestCase):
    """Test FilterError exception class."""

    def test_filter_error_creation(self) -> None:
        """Test FilterError can be created with a value."""
        value = 42
        error = FilterError(value)

        assert isinstance(error, Exception)
        assert isinstance(error, FilterError)
        assert error.value == value

    def test_filter_error_message(self) -> None:
        """Test FilterError has the correct error message."""
        value = "test_value"
        error = FilterError(value)

        expected_message = f"Value {value} did not pass filter predicate"
        assert str(error) == expected_message

    def test_filter_error_with_none_value(self) -> None:
        """Test FilterError with None value."""
        error = FilterError(None)

        assert error.value is None
        assert str(error) == "Value None did not pass filter predicate"

    def test_filter_error_with_complex_value(self) -> None:
        """Test FilterError with complex object."""
        value = {"key": "value", "number": 123}
        error = FilterError(value)

        assert error.value == value
        expected_message = f"Value {value} did not pass filter predicate"
        assert str(error) == expected_message

    def test_filter_error_with_list_value(self) -> None:
        """Test FilterError with list value."""
        value = [1, 2, 3, "test"]
        error = FilterError(value)

        assert error.value == value
        expected_message = f"Value {value} did not pass filter predicate"
        assert str(error) == expected_message

    def test_filter_error_inheritance(self) -> None:
        """Test FilterError properly inherits from Exception."""
        error = FilterError("test")

        assert isinstance(error, Exception)
        assert issubclass(FilterError, Exception)

    def test_filter_error_can_be_raised(self) -> None:
        """Test FilterError can be raised and caught."""
        value = "failing_value"

        with pytest.raises(FilterError) as context:
            raise FilterError(value)

        assert context.value.value == value
        assert str(context.value) == f"Value {value} did not pass filter predicate"

    def test_filter_error_with_zero(self) -> None:
        """Test FilterError with zero value."""
        error = FilterError(0)

        assert error.value == 0
        assert str(error) == "Value 0 did not pass filter predicate"

    def test_filter_error_with_empty_string(self) -> None:
        """Test FilterError with empty string."""
        error = FilterError("")

        assert error.value == ""
        assert str(error) == "Value  did not pass filter predicate"

    def test_filter_error_with_boolean(self) -> None:
        """Test FilterError with boolean values."""
        # Test with False
        error_false = FilterError(value=False)
        assert error_false.value is False
        assert str(error_false) == "Value False did not pass filter predicate"

        # Test with True
        error_true = FilterError(value=True)
        assert error_true.value is True
        assert str(error_true) == "Value True did not pass filter predicate"

    def test_filter_error_value_access(self) -> None:
        """Test that the original value can be accessed from the exception."""
        original_value = {"data": [1, 2, 3], "status": "failed"}
        error = FilterError(original_value)

        # The value should be exactly the same object
        assert error.value == original_value
        assert error.value["data"] == [1, 2, 3]
        assert error.value["status"] == "failed"


if __name__ == "__main__":
    unittest.main()
