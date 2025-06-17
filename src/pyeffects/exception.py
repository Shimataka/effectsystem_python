from typing import Any


class FilterError(Exception):
    """filter で条件に合わない場合のエラー

    Args:
        value (Any): The value that did not pass the filter predicate.
    """

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        self.value = value
        super().__init__(f"Value {value} did not pass filter predicate")
