"""Слой 2: сборка матрицы признаков конкретного эксперимента из кандидатов слоя 1
(применение d_features.resolution, вычет forbidden_as_features, extra_features)."""

import pandas as pd

from src.families import expand_feature_list, get_forbidden_columns


def _dedup_minus_forbidden(columns: list[str], forbidden: set[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for col in columns:
        if col in forbidden or col in seen:
            continue
        seen.add(col)
        result.append(col)
    return result


def model_feature_columns(config: dict) -> tuple[list[str], list[str]]:
    features = config["features"]
    forbidden = set(get_forbidden_columns(config))
    extra = config.get("extra_features") or {}

    numeric = expand_feature_list(features["numeric"], config)
    categorical = expand_feature_list(features["categorical"], config)

    for col, resolution in config["d_features"]["resolution"].items():
        if resolution == "raw":
            numeric.append(col)
        elif resolution == "norm":
            numeric.append(col + "Norm")
        elif resolution != "excluded":
            raise ValueError(f"{col}: неизвестная резолюция {resolution!r}")

    numeric.extend(extra.get("numeric", []))
    categorical.extend(extra.get("categorical", []))

    return _dedup_minus_forbidden(numeric, forbidden), _dedup_minus_forbidden(categorical, forbidden)


def _cast_int_like_float(s: pd.Series) -> pd.Series:
    """card2, card3, card5, addr1, addr2, id_13, ... — float64 с NaN, хотя по смыслу
    целочисленные коды. Прямой astype(str) даёт '1234.0'; здесь непустые значения
    сперва приводятся к nullable Int64, если колонка целочисленна по смыслу."""
    if s.dtype.kind != "f":
        return s
    non_null = s.dropna()
    if non_null.empty or not (non_null % 1 == 0).all():
        return s
    return s.astype("Int64")


def normalize_categorical_column(s: pd.Series, fill_value: str) -> pd.Series:
    """int (где применимо) -> fillna(fill_value) -> str, с гарантированным dtype
    object без NaN на выходе (астype(str) в pandas 3 с infer_string даёт расширение
    "str", а не object — поэтому явный map(str) + astype(object))."""
    s = _cast_int_like_float(s)
    filled = s.astype(object).where(s.notna(), fill_value)
    return filled.map(str).astype(object)


def build_matrix(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    numeric, categorical = model_feature_columns(config)
    fill_value = config["features"]["categorical_fill"]

    X = df[numeric + categorical].copy()
    for col in categorical:
        X[col] = normalize_categorical_column(X[col], fill_value)

    return X, categorical
