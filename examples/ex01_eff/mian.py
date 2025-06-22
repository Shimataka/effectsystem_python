# ruff: noqa: BLE001, T201, S311

import random

from pyresults import Err, Ok

from pyeffects import Eff, FnEff


def example01() -> None:
    print("1. Effectの実行と作成")

    # 関数をEffectに変換
    def risky_operation() -> int:
        if random.random() < 0.5:
            msg = "Something went wrong"
            raise ValueError(msg)
        return 42

    effect1 = FnEff[int, ValueError](risky_operation)

    # 実行方法1: unwrap() - 例外が発生する可能性
    try:
        result = effect1.unwrap()
        print(f"Success: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # 実行方法2: run() - Resultで包まれて返される
    result = effect1.run()
    match result:
        case Ok(value):
            print(f"Success: {value}")
        case Err(error):
            print(f"Error: {error}")
        case _:
            print("Not reachable")


def example02() -> None:
    print("2. メソッドチェーン")

    # 複雑な処理をチェーンで組み立て
    effect2 = (
        FnEff[str, ValueError](lambda: "hello world")
        .map(str.upper)  # "HELLO WORLD"
        .filter(lambda s: len(s) > 5)  # 長さチェック
        .tap(lambda s: print(f"Processing: {s}"))  # ログ出力
        .recover(lambda _: "DEFAULT")
    )  # エラー時のデフォルト値

    print("Not run effect2 yet")
    result = effect2.run()  # "HELLO WORLD"
    print(f"run effect2: {result}")


def example03() -> None:
    print("3. エラーハンドリング")

    def unreliable_api_call() -> str:
        import random

        if random.random() < 0.7:  # 70%の確率で失敗
            msg = "API unavailable"
            raise ConnectionError(msg)
        return "API response data"

    # 堅牢なエラーハンドリング
    safe_effect = (
        FnEff[str, ConnectionError](unreliable_api_call)
        .retry(max_retries=3, delay=1.0)  # 3回リトライ
        .timeout(5.0)  # 5秒でタイムアウト
        .recover(lambda _: "Fallback data")
    )  # エラー時のフォールバック

    result = safe_effect.run()
    print(f"run recovered effect: {result}")


def example04() -> None:
    print("4. 並列処理")

    # 複数のAPI呼び出しを並列実行
    def fetch_user_data(user_id: int) -> list[str]:
        return [f"User data for {user_id}"]

    effects: list[Eff[list[str], Exception]] = [
        FnEff[list[str], Exception](lambda: fetch_user_data(1)),
        FnEff[list[str], Exception](lambda: fetch_user_data(2)),
        FnEff[list[str], Exception](lambda: fetch_user_data(3)),
    ]

    # 全て完了を待つ
    all_results = Eff[list[str], Exception].parallel(effects).run()
    print(f"Retrieved {len(all_results.unwrap_or([]))} users")

    # 最初の完了を待つ
    first_result = Eff[list[str], Exception].race(effects).run()
    print(f"First result: {first_result.unwrap_or([])}")


if __name__ == "__main__":
    example01()
    example02()
    example03()
    example04()
