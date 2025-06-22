# ruff: noqa: BLE001, T201, S311

import json
from pathlib import Path

from pyresults import Err, Ok

from pyeffects import FnEff


def read_config_file(path: Path) -> dict[str, str]:
    with path.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def validate_config(config: dict[str, str]) -> dict[str, str]:
    required_keys = ["api_key", "database_url"]
    if not all(key in config for key in required_keys):
        msg = "Missing required configuration keys"
        raise ValueError(msg)
    return config


# 設定ファイル読み込みパイプライン
config_effect = (
    FnEff[dict[str, str], ValueError](lambda: read_config_file(Path("config.json")))
    .flat_map(lambda config: FnEff[dict[str, str], ValueError](lambda: validate_config(config)))
    .tap(lambda _: print("Configuration loaded successfully"))
    .recover(lambda _: {"api_key": "default", "database_url": "sqlite:///default.db"})
)

# Result型で安全に取得
config_result = config_effect.run()
match config_result:
    case Ok(config):
        print(f"Using config: {config}")
    case Err(error):
        print(f"Failed to load config: {error}")
    case _:
        print("Not reachable")
