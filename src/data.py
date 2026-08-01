import pandas as pd

_VALID_SPLITS = ("train", "test")


def _normalize_test_identity_columns(identity_df: pd.DataFrame) -> pd.DataFrame:
    """test_identity.csv в IEEE-CIS называет id-колонки через дефис (id-01),
    а не подчёркивание (id_01), как в train_identity.csv. Приводим к единому виду перед merge."""
    return identity_df.rename(columns=lambda c: c.replace("id-", "id_", 1) if c.startswith("id-") else c)


def merge(transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
    """Left join по TransactionID через pd.concat, а не pd.merge: merge на df с
    сотнями разнотипных колонок сам по себе фрагментирует внутренние блоки pandas
    (PerformanceWarning) - reindex+concat строит результат одним блочным проходом."""
    identity_aligned = identity_df.set_index("TransactionID").reindex(transaction_df["TransactionID"].values)
    identity_aligned.index = transaction_df.index
    has_identity = transaction_df["TransactionID"].isin(identity_df["TransactionID"]).astype(int).rename("hasIdentity")
    return pd.concat([transaction_df, identity_aligned, has_identity], axis=1)


def sort_by_transaction_dt(df: pd.DataFrame) -> pd.DataFrame:
    """has_time=True (CatBoost) требует строки, отсортированные по времени.
    mergesort — стабильная сортировка: одновременные транзакции сохраняют исходный
    порядок, а не переставляются произвольно."""
    return df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)


def load_split(config: dict, split: str) -> pd.DataFrame:
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

    paths = config["paths"]
    transaction_df = pd.read_csv(paths[f"{split}_transaction"])
    identity_df = pd.read_csv(paths[f"{split}_identity"])
    if split == "test":
        identity_df = _normalize_test_identity_columns(identity_df)

    return merge(transaction_df, identity_df)
