# common_patterns

頻繁に利用するコマンドパターンや典型的な実装テンプレートなど

## 開発ワークフローでのTIPS

### Python開発

- **パッケージの管理**: `uv`
- **コード実行**: `uv run src/{package_name}`  (ディレクトリを指定しても実行可能)
- **テスト**: `pytest`
- **テストの書き方**: 関数には必ず型ヒントを書き、テストコードは`tests`ディレクトリに配置する。
- **フォーマット**: `uvx ruff format src tests`
- **リント**: `uvx ruff lint src tests`
- **型チェック**: `uvx mypy src tests`

### Rust開発

- **ビルド**: `cargo build`
- **実行**: `cargo run`
- **テスト**: `cargo test`
- **ドキュメント**: `cargo doc`

## コミットメッセージ要件

このリポジトリはpre-commitフックとCIを通じて[Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/)を強制します：

形式: `<type>: <description>`

有効なタイプ: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `perf`, `refactor`, `style`, `test`
