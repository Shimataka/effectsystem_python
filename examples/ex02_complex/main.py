# ruff: noqa: BLE001, T201, S311

from pyresults import Err, Ok

from pyeffects import FnEff


def safe_division(a: int, b: int) -> int:
    if b == 0:
        msg = "Division by zero"
        raise ZeroDivisionError(msg)
    return a // b


effect = FnEff[int, ZeroDivisionError](lambda: safe_division(10, 0))

# Option 1: 例外ベース
try:
    result = effect.unwrap()
except ZeroDivisionError:
    print("Cannot divide by zero")

# Option 2: Result型ベース
result = effect.run()
match result:
    case Ok(value):
        print(f"Division result: {value}")
    case Err(ZeroDivisionError()):
        print("Cannot divide by zero")
    case Err(error):
        print(f"Unexpected error: {error}")
    case _:
        print("Not reachable")
