"""Схема валидации (02_feature_eda, §6): временной holdout + скользящий CV по месячным
блокам. Идентична во всех экспериментах, параметры — из configs/base.yaml."""

import warnings

import pandas as pd

Split = tuple[pd.Index, pd.Index]


def holdout_split(df: pd.DataFrame, config: dict) -> Split:
    quantile = config["validation"]["holdout_quantile"]
    split_day = df["TransactionDay"].quantile(quantile)
    train_idx = df.index[df["TransactionDay"] < split_day]
    valid_idx = df.index[df["TransactionDay"] >= split_day]
    return train_idx, valid_idx


def cv_folds(df: pd.DataFrame, config: dict) -> list[Split]:
    validation = config["validation"]
    block_days = validation["month_block_days"]
    skip_first = validation["cv_skip_first_blocks"]
    min_valid_rows = validation["min_valid_rows"]

    month_block = df["TransactionDay"] // block_days
    blocks = sorted(month_block.unique())

    if validation["drop_incomplete_last_block"]:
        last_block_days = df.loc[month_block == blocks[-1], "TransactionDay"].nunique()
        if last_block_days < block_days:
            blocks = blocks[:-1]

    folds = []
    for b in blocks[skip_first:-1]:
        train_idx = df.index[month_block <= b]
        valid_idx = df.index[month_block == b + 1]
        if len(valid_idx) < min_valid_rows:
            warnings.warn(
                f"cv_folds: фолд b={b} отброшен — valid_rows={len(valid_idx)} < min_valid_rows={min_valid_rows}"
            )
            continue
        folds.append((train_idx, valid_idx))
    return folds
