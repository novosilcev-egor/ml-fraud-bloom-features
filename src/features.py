import re

import numpy as np
import pandas as pd

from src.families import get_d_columns

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = pd.DataFrame(
        {
            "TransactionAmtLog": np.log1p(df["TransactionAmt"]),
            "TransactionNDecimal": df["TransactionAmt"].apply(
                lambda x: len(str(x).split(".")[1].rstrip("0")) if "." in str(x) else 0
            ),
        }
    )
    return pd.concat([df, new_cols], axis=1)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = (df["TransactionDT"] // SECONDS_PER_HOUR) % HOURS_PER_DAY
    day = df["TransactionDT"] // SECONDS_PER_DAY
    dow = (day % DAYS_PER_WEEK).astype(int)
    new_cols = pd.DataFrame(
        {
            "TransactionHour": hour,
            "TransactionHourCat": hour.astype(int).map(str).astype(object),
            "TransactionDay": day,
            "TransactionDoW": dow,
        }
    )
    return pd.concat([df, new_cols], axis=1)


def add_d_norm_features(df: pd.DataFrame, d_columns: list[str]) -> pd.DataFrame:
    new_cols = pd.DataFrame({col + "Norm": df[col] - df["TransactionDay"] for col in d_columns})
    return pd.concat([df, new_cols], axis=1)


def _email_suffix(domain):
    if pd.isna(domain):
        return "missing"
    parts = domain.split(".", 1)
    return parts[1] if len(parts) > 1 else "none"


def _email_provider_group(domain, provider_groups: dict):
    if pd.isna(domain):
        return "missing"
    provider = domain.split(".")[0].lower()
    return provider_groups.get(provider, provider)


def add_email_features(df: pd.DataFrame, provider_groups: dict) -> pd.DataFrame:
    new_cols = {}
    for col in ("P_emaildomain", "R_emaildomain"):
        prefix = col.split("_")[0]
        new_cols[prefix + "_EmailSuffix"] = df[col].apply(_email_suffix)
        new_cols[prefix + "_EmailProviderGroup"] = df[col].apply(
            lambda domain: _email_provider_group(domain, provider_groups)
        )
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def _device_family(info, device_rules: list[tuple[str, str]]):
    if pd.isna(info):
        return "missing"
    low = str(info).lower()
    for pattern, family in device_rules:
        if pattern in low:
            return family
    return "other"


def add_device_family(df: pd.DataFrame, device_rules: list[tuple[str, str]]) -> pd.DataFrame:
    new_col = df["DeviceInfo"].apply(lambda info: _device_family(info, device_rules)).rename("DeviceFamily")
    return pd.concat([df, new_col], axis=1)


def _browser_family(val):
    if pd.isna(val):
        return "missing"
    return re.sub(r"[\d.]+", "", str(val)).strip() or "other"


def add_browser_family(df: pd.DataFrame) -> pd.DataFrame:
    new_col = df["id_31"].apply(_browser_family).rename("BrowserFamily")
    return pd.concat([df, new_col], axis=1)


def build_candidate_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    derived = config["derived"]
    device_rules = [tuple(rule) for rule in derived["device_rules"]]

    df = add_amount_features(df)
    df = add_time_features(df)
    df = add_d_norm_features(df, get_d_columns(config))
    df = add_email_features(df, derived["provider_groups"])
    df = add_device_family(df, device_rules)
    df = add_browser_family(df)
    return df
