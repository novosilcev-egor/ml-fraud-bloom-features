from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


def _fit_and_score(train_features: pd.DataFrame, test_features: pd.DataFrame, model_params: dict) -> float:
    X = pd.concat([train_features, test_features], ignore_index=True)
    y = [0] * len(train_features) + [1] * len(test_features)

    model = CatBoostClassifier(**model_params, verbose=False, allow_writing_files=False)
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, preds)


def run_av(train_df: pd.DataFrame, test_df: pd.DataFrame, d_columns: list[str], config: dict) -> pd.DataFrame:
    av_config = config["adversarial_validation"]
    model_params = av_config["model_params"]
    threshold = av_config["auc_pass_threshold"]

    rows = []
    for col in d_columns:
        for version, colname in (("raw", col), ("norm", col + "Norm")):
            auc = _fit_and_score(train_df[[colname]], test_df[[colname]], model_params)
            verdict = "pass" if auc <= threshold else "fail"
            rows.append({"column": col, "version": version, "auc": auc, "verdict": verdict})

    return pd.DataFrame(rows)


def summarize_d_block(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    d_columns: list[str],
    av_table: pd.DataFrame,
    config: dict,
) -> dict:
    model_params = config["adversarial_validation"]["model_params"]

    auc_before = _fit_and_score(train_df[d_columns], test_df[d_columns], model_params)

    selected_columns = []
    for col in d_columns:
        passing = av_table[(av_table["column"] == col) & (av_table["verdict"] == "pass")]
        if passing.empty:
            continue
        best = passing.loc[passing["auc"].idxmin()]
        selected_columns.append(col if best["version"] == "raw" else col + "Norm")

    if selected_columns:
        auc_after = _fit_and_score(train_df[selected_columns], test_df[selected_columns], model_params)
    else:
        auc_after = float("nan")

    return {
        "auc_before": auc_before,
        "auc_after": auc_after,
        "n_columns_before": len(d_columns),
        "n_columns_after": len(selected_columns),
        "columns_after": selected_columns,
    }


def save_av_table(av_table: pd.DataFrame, path: str | Path) -> None:
    av_table.to_csv(path, index=False)


def save_av_summary(summary: dict, path: str | Path) -> None:
    row = {k: v for k, v in summary.items() if k != "columns_after"}
    row["columns_after"] = ",".join(summary["columns_after"])
    pd.DataFrame([row]).to_csv(path, index=False)
