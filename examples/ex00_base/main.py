# ruff: noqa: T201, BLE001

from pathlib import Path

from pyeffects import Eff, Effect
from pyeffects.deco import effect, io
from pyeffects.example import NetworkEffect

if __name__ == "__main__":
    # 1. 関数ベースの使用例
    def read_file(path: Path) -> Eff[str]:
        with path.open() as f:
            return Eff(f.read())

    @effect
    def write_file(path: Path, content: str) -> None:
        with path.open("w") as f:
            f.write(content)

    @io
    def print_message(message: str) -> None:
        print(f"[LOG] {message}")

    # 2. エフェクトの組み合わせ
    def process_file_content(input_path: Path, output_path: Path) -> Effect[None]:
        return (
            read_file(input_path)
            .tap(lambda content: print(f"Read {len(content)} characters"))
            .map(str.upper)
            .tap(lambda _: print("Converted to uppercase"))
            .flat_map(lambda content: write_file(output_path, content))
            .recover(lambda e: print(f"Error: {e}"))
        )

    # 3. 並列処理の例
    def parallel_file_processing(paths: list[Path]) -> Effect[list[str]]:
        read_effects: list[Effect[str]] = [read_file(path) for path in paths]

        def func(contents: list[str]) -> list[str]:
            return [c.strip() for c in contents]

        return Effect[str].parallel(read_effects).map(func)

    # 4. リトライ付きネットワーク通信
    def robust_api_call(url: str) -> Effect[str]:
        return (
            NetworkEffect(url)
            .retry(max_retries=3, delay=1.0)
            .timeout(10.0)
            .recover(lambda e: f"Failed to fetch {url}: {e}")
        )

    # 5. 複雑なエフェクトチェーン
    def complex_workflow(config_path: Path) -> Effect[dict[str, str]]:
        return (
            read_file(config_path)
            .filter(lambda content: len(content) > 0)
            .map(lambda content: {"config": content, "processed_at": "2025-06-16"})
            .tap(lambda result: print(f"Processed config: {result}"))
            .memorize()
        )  # 結果をキャッシュ

    # 実行例
    try:
        # 単純な実行
        result = read_file(Path("example.txt")).unwrap()  # <- ここで実行される (遅延)
        print(f"File content: {result}")

        # エフェクトチェーンの実行
        workflow_result = complex_workflow(
            Path("config.json"),
        ).unwrap()  # <- ここで実行される (遅延)
        print(f"Workflow result: {workflow_result}")

    except Exception as e:
        print(f"Error occurred: {e}")
