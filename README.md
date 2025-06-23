# PyEffects

Pythonで副作用を制御可能にする関数型プログラミングライブラリ

## 概要

PyEffectsは、Pythonにおいて副作用（IO、ネットワーク通信、データベースアクセスなど）を型安全かつ合成可能な方法で扱うためのライブラリです。HaskellのIOモナドやScalaのEffectシステムにインスパイアされ、遅延実行とメソッドチェーンによる直感的なAPIを提供します。

## 特徴

- **🚀 遅延実行**: 副作用は`unwrap()`が呼ばれるまで実行されません
- **🔗 メソッドチェーン**: 関数型操作による流暢なプログラミング体験
- **🛡️ 型安全性**: TypeHintsによる完全な型サポート
- **🔄 エラーハンドリング**: `recover`と`retry`による堅牢なエラー処理
- **⚡ 並列処理**: `parallel`と`race`による効率的な並行実行
- **🧪 テスト容易性**: 遅延実行により簡単なモック化が可能
- **🔀 ハイブリッド設計**: 関数ベースとクラスベースの両方をサポート

## インストール

```bash
pip install git+https://github.com/Shimataka/effectsystem_python.git
```

## クイックスタート

### 基本的な使用方法

```python
from pyeffects import effect, Effect, Eff
from pathlib import Path

# 関数をエフェクトに変換
def read_file(path: Path) -> Eff[str]:
    with path.open() as f:
        return Eff(f.read())

@effect
def write_file(path: Path, content: str) -> None:
    with path.open("w") as f:
        f.write(content)

# エフェクトの合成と実行
result = (read_file(Path("input.txt"))
          .map(str.upper)                    # 大文字変換
          .filter(lambda s: len(s) > 0)      # 空文字列をフィルタ
          .tap(lambda s: print(f"処理中: {s[:10]}..."))  # ログ出力
          .unwrap())                         # 実行

print(f"結果: {result}")
```

### エラーハンドリング

```python
# 堅牢なエラーハンドリング
safe_result = (read_file(Path("config.json"))
               .retry(max_attempts=3, delay=1.0)    # 3回リトライ
               .timeout(5.0)                        # 5秒でタイムアウト
               .recover(lambda e: "{}"))            # エラー時はデフォルト値
               .unwrap())
```

### 並列処理

```python
# 複数のファイルを並列処理
files = [Path(f"file{i}.txt") for i in range(5)]
read_effects = [read_file(path) for path in files]

all_contents = Effect.parallel(read_effects).unwrap()
print(f"読み込んだファイル数: {len(all_contents)}")
```

## 主要なメソッド

- 変換操作

    | メソッド | 説明 | 例 |
    |---------|------|-----|
    | `map(func)` | 結果を変換 | `.map(str.upper)` |
    | `flat_map(func)` | エフェクトチェーン | `.flat_map(lambda x: other_effect(x))` |
    | `filter(predicate)` | 条件フィルタ（例外版） | `.filter(lambda x: x > 0)` |
    | `find(predicate)` | 条件検索（Optional版） | `.find(lambda x: x.startswith("prefix"))` |

- エラーハンドリング

    | メソッド | 説明 | 例 |
    |---------|------|-----|
    | `recover(handler)` | エラー回復 | `.recover(lambda e: "default")` |
    | `retry(max_attempts, delay)` | リトライ | `.retry(3, 1.0)` |
    | `timeout(seconds)` | タイムアウト | `.timeout(10.0)` |

- 副作用操作

    | メソッド | 説明 | 例 |
    |---------|------|-----|
    | `tap(side_effect)` | 副作用実行（値は変更しない） | `.tap(print)` |
    | `memoize()` | 結果をキャッシュ | `.memoize()` |

- 組み合わせ操作

    | メソッド | 説明 | 例 |
    |---------|------|-----|
    | `zip_with(other, combiner)` | 2つのエフェクトを結合 | `.zip_with(other, lambda x, y: x + y)` |
    | `Effect.sequence(effects)` | 順次実行 | `Effect.sequence([eff1, eff2, eff3])` |
    | `Effect.parallel(effects)` | 並列実行 | `Effect.parallel([eff1, eff2, eff3])` |
    | `Effect.race(effects)` | 最初の完了を待つ | `Effect.race([eff1, eff2])` |

## 高度な使用例

### カスタムエフェクトクラス

```python
class DatabaseEffect(Effect[list[dict]]):
    """データベース操作用の特殊なエフェクト"""

    def __init__(self, query: str, params: dict = None):
        self.query = query
        self.params = params or {}

    def unwrap(self) -> list[dict]:
        # データベース接続とクエリ実行
        connection = get_db_connection()
        return connection.execute(self.query, self.params)

# 使用例
users = (DatabaseEffect("SELECT * FROM users WHERE age > ?", {"age": 18})
         .filter(lambda results: len(results) > 0)
         .map(lambda results: [user["name"] for user in results])
         .unwrap())
```

### 複雑なワークフロー

```python
def data_processing_pipeline(input_path: Path, output_path: Path) -> Effect[str]:
    """データ処理パイプライン"""
    return (read_file(input_path)
            .filter(lambda content: len(content.strip()) > 0)
            .map(parse_csv_data)
            .map(clean_data)
            .map(transform_data)
            .tap(lambda data: print(f"処理済みレコード数: {len(data)}"))
            .map(serialize_data)
            .flat_map(lambda data: write_file(output_path, data))
            .map(lambda _: f"パイプライン完了: {output_path}")
            .recover(lambda e: f"パイプラインエラー: {e}"))

# 実行
result = data_processing_pipeline(
    Path("input.csv"),
    Path("output.json")
).unwrap()
```

### 条件分岐とエラーハンドリング

```python
def robust_api_workflow(user_id: int) -> Effect[dict]:
    """API呼び出しの堅牢なワークフロー"""
    return (fetch_user_data(user_id)
            .retry(max_attempts=3, delay=2.0)
            .filter(lambda user: user.get("active", False))
            .flat_map(lambda user: fetch_user_preferences(user["id"]))
            .zip_with(
                fetch_user_permissions(user_id),
                lambda prefs, perms: {"preferences": prefs, "permissions": perms}
            )
            .timeout(30.0)
            .recover(lambda e: {"error": str(e), "user_id": user_id}))
```

## 設計パターン

### 1. 関数ベース（シンプルな副作用）

```python
@effect
def simple_operation(x: int) -> int:
    # シンプルな処理
    return x * 2
```

### 2. クラスベース（複雑な副作用）

```python
class ComplexEffect(Effect[ResultType]):
    def __init__(self, config: dict):
        self.config = config

    def unwrap(self) -> ResultType:
        # 複雑な処理ロジック
        pass
```

### 3. ハイブリッドアプローチ

```python
# 基本操作は関数ベース
@effect
def basic_io(path: Path) -> str:
    return path.read_text()

# 複雑な操作はクラスベース
class AdvancedProcessor(Effect[ProcessedData]):
    # 複雑な状態管理と処理
    pass

# 組み合わせて使用
result = (basic_io(config_path)
          .flat_map(lambda config: AdvancedProcessor(parse_config(config)))
          .unwrap())
```

## テストでの活用

```python
import unittest
from unittest.mock import patch

class TestEffects(unittest.TestCase):
    def test_file_processing(self):
        # エフェクトは遅延実行なのでモック化が簡単
        with patch('pathlib.Path.open') as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "test data"

            effect = read_file(Path("test.txt"))
            result = effect.unwrap()

            self.assertEqual(result, "test data")
            mock_open.assert_called_once()
```

## パフォーマンスの考慮事項

- **遅延実行**: 不要な計算を避けるため、条件分岐と組み合わせて効果的
- **メモ化**: `memoize()`で重い処理結果をキャッシュ
- **並列処理**: `parallel()`でI/Oバウンドなタスクを高速化
- **タイムアウト**: `timeout()`で長時間実行を防止

## よくある使用パターン

### ファイル処理

```python
# 設定ファイルの読み込みと検証
config = (read_file(Path("config.json"))
          .map(json.loads)
          .filter(lambda cfg: "api_key" in cfg)
          .recover(lambda e: {"api_key": "default"})
          .unwrap())
```

### API呼び出し

```python
# 外部API呼び出しの堅牢化
response = (http_get("https://api.example.com/data")
            .retry(3, 2.0)
            .timeout(10.0)
            .map(json.loads)
            .filter(lambda data: data.get("status") == "success")
            .unwrap())
```

### データベース操作

```python
# データベースからの安全なデータ取得
users = (DatabaseQuery("SELECT * FROM users WHERE active = 1")
         .retry(2)
         .filter(lambda results: len(results) > 0)
         .map(lambda results: [User.from_dict(row) for row in results])
         .recover(lambda e: [])
         .unwrap())
```

## ライセンス

MIT License

## 関連リンク

- [使用例集](examples/)
