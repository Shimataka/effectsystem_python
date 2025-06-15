from typing import Any


class FilterError(Exception):
    """filter で条件に合わない場合のエラー"""

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        self.value = value
        super().__init__(f"Value {value} did not pass filter predicate")
