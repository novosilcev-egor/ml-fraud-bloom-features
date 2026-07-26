from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "configs/base.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
