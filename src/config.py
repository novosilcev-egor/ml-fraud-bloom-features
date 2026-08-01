from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "configs/base.yaml"

_EXPERIMENT_ALLOWED_KEYS = {"extends", "experiment", "extra_features"}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: str | Path) -> dict:
    """base.yaml + конфиг эксперимента. Файл эксперимента не может переопределять
    ничего, кроме extra_features."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        experiment_config = yaml.safe_load(f)

    extra_keys = set(experiment_config) - _EXPERIMENT_ALLOWED_KEYS
    if extra_keys:
        raise ValueError(
            f"{path}: файл эксперимента переопределяет запрещённые ключи {sorted(extra_keys)}; "
            f"разрешены только {sorted(_EXPERIMENT_ALLOWED_KEYS)}"
        )

    config = load_config(path.parent / experiment_config["extends"])
    config["experiment"] = experiment_config["experiment"]
    config["extra_features"] = experiment_config.get("extra_features", {})
    return config
