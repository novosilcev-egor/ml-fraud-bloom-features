"""Слой 2 + пайплайн эксперимента: holdout, скользящий CV, переобучение на всём
train и сабмит (02_feature_eda, §7, задача 2). Стадийный (--stage), чтобы прогон
можно было продолжить после остановки: состояние по run_id — в
reports/runs/<run_id>/metrics.json; сабмит — в reports/submissions/<run_id>.csv;
итоговая строка — в reports/experiments.md (на стадии submit, либо отдельно на стадии
log — если нужно пересчитать только cv для существующего run_id, не трогая holdout/submit).
"""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from src.config import load_experiment_config
from src.data import load_split, sort_by_transaction_dt
from src.features import build_candidate_features
from src.matrix import build_matrix
from src.splits import cv_folds, holdout_split

STAGES = ("holdout", "cv", "submit", "log", "all")


def _peak_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def _run_id(experiment: str) -> str:
    return f"{experiment}_{datetime.now():%Y%m%d_%H%M%S}"


def _run_dir(config: dict, run_id: str) -> Path:
    return Path(config["paths"]["runs_dir"]) / run_id


def _load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_metrics(run_dir: Path, metrics: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)


def _check_av_gate(config: dict) -> None:
    status = config["d_features"]["status"]
    if status != "applied_from_av":
        sys.exit(
            f"d_features.status = {status!r}, ожидается 'applied_from_av'. "
            "AV не пересчитывается автоматически — сверьтесь с reports/av_d_features.csv "
            "и впишите резолюцию в configs/base.yaml вручную."
        )


def _load_train_matrix(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    df = load_split(config, "train")
    df = sort_by_transaction_dt(df)
    df = build_candidate_features(df, config)
    X, cat_features = build_matrix(df, config)
    y = df[config["features"]["target"]]
    return df, X, cat_features, y


def _model_params(config: dict, iterations: int | None = None, early_stopping: bool = True) -> dict:
    params = dict(config["model"]["params"])
    if iterations is not None:
        params["iterations"] = iterations
    if not early_stopping:
        params.pop("early_stopping_rounds", None)
    return params


def _fit(X_train, y_train, cat_features, params, task_type, eval_set=None):
    model = CatBoostClassifier(**params, task_type=task_type, verbose=False, allow_writing_files=False)
    pool_train = Pool(X_train, y_train, cat_features=cat_features)
    pool_eval = Pool(*eval_set, cat_features=cat_features) if eval_set is not None else None

    t0 = time.perf_counter()
    model.fit(pool_train, eval_set=pool_eval, use_best_model=pool_eval is not None)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def run_holdout(config: dict, run_id: str) -> dict:
    run_dir = _run_dir(config, run_id)
    metrics = _load_metrics(run_dir)

    df, X, cat_features, y = _load_train_matrix(config)
    train_idx, valid_idx = holdout_split(df, config)

    params = _model_params(config, early_stopping=True)
    model, elapsed = _fit(
        X.loc[train_idx],
        y.loc[train_idx],
        cat_features,
        params,
        config["model"]["task_type"],
        eval_set=(X.loc[valid_idx], y.loc[valid_idx]),
    )
    preds = model.predict_proba(Pool(X.loc[valid_idx], cat_features=cat_features))[:, 1]
    auc = roc_auc_score(y.loc[valid_idx], preds)

    metrics["holdout"] = {
        "auc": auc,
        "best_iteration": model.get_best_iteration(),
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "train_max_transaction_day": float(df.loc[train_idx, "TransactionDay"].max()),
        "valid_min_transaction_day": float(df.loc[valid_idx, "TransactionDay"].min()),
        "seconds": elapsed,
        "peak_memory_bytes": _peak_memory_bytes(),
    }
    _save_metrics(run_dir, metrics)

    print(f"run_id: {run_id}")
    print(f"[holdout] AUC={auc:.5f} best_iteration={metrics['holdout']['best_iteration']} time={elapsed:.1f}s")
    return metrics


def run_cv(config: dict, run_id: str) -> dict:
    run_dir = _run_dir(config, run_id)
    metrics = _load_metrics(run_dir)

    df, X, cat_features, y = _load_train_matrix(config)
    folds = cv_folds(df, config)

    params = _model_params(config, early_stopping=True)
    fold_results = []
    total_seconds = 0.0
    for i, (train_idx, valid_idx) in enumerate(folds):
        model, elapsed = _fit(
            X.loc[train_idx],
            y.loc[train_idx],
            cat_features,
            params,
            config["model"]["task_type"],
            eval_set=(X.loc[valid_idx], y.loc[valid_idx]),
        )
        preds = model.predict_proba(Pool(X.loc[valid_idx], cat_features=cat_features))[:, 1]
        auc = roc_auc_score(y.loc[valid_idx], preds)
        total_seconds += elapsed
        fold_results.append(
            {
                "fold": i,
                "auc": auc,
                "best_iteration": model.get_best_iteration(),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "seconds": elapsed,
            }
        )
        print(f"[cv fold {i}] AUC={auc:.5f} time={elapsed:.1f}s")

    aucs = pd.Series([r["auc"] for r in fold_results])
    metrics["cv"] = {
        "folds": fold_results,
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "seconds": total_seconds,
        "peak_memory_bytes": _peak_memory_bytes(),
    }
    _save_metrics(run_dir, metrics)

    print(f"[cv] mean AUC={metrics['cv']['auc_mean']:.5f} ± {metrics['cv']['auc_std']:.5f}")
    return metrics


def _git_commit_hash() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _git_hash_object(path: Path) -> str:
    result = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True)
    return result.stdout.strip()[:12]


def _base_config_path(experiment_config_path: Path) -> Path:
    with open(experiment_config_path, encoding="utf-8") as f:
        experiment_yaml = yaml.safe_load(f)
    return experiment_config_path.parent / experiment_yaml["extends"]


def _append_experiments_log(config: dict, run_id: str, metrics: dict, config_path: Path) -> None:
    log_path = Path(config["paths"]["experiments_log"])
    header = (
        "| Дата | run_id | Эксперимент | Git (репо/base/эксп.) | Сплиты | Holdout AUC | "
        "CV-mean AUC (mean±std) | per-CV AUC (test size) | Время, с | Пик. память, МиБ |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not log_path.exists() or log_path.stat().st_size == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("# Журнал экспериментов\n\n" + header, encoding="utf-8")

    holdout = metrics["holdout"]
    cv = metrics.get("cv")
    submit = metrics["submit"]

    splits = f"holdout: valid day≥{holdout['valid_min_transaction_day']:.0f}"
    if cv:
        splits += f"; CV: {len(cv['folds'])} фолдов"

    total_seconds = holdout["seconds"] + (cv["seconds"] if cv else 0.0) + submit["seconds"]
    peak_memory_mib = (
        max(holdout["peak_memory_bytes"], cv["peak_memory_bytes"] if cv else 0, submit["peak_memory_bytes"])
        / 1024**2
    )
    cv_mean_repr = f"{cv['auc_mean']:.5f} ± {cv['auc_std']:.5f}" if cv else "—"
    cv_per_fold_repr = " <br> ".join(f"{r['auc']:.5f} ({r['valid_rows']})" for r in cv["folds"]) if cv else "—"

    row = (
        f"| {datetime.now():%Y-%m-%d %H:%M} | {run_id} | {config['experiment']} "
        f"| {_git_commit_hash()}/{_git_hash_object(_base_config_path(config_path))}/{_git_hash_object(config_path)} "
        f"| {splits} | {holdout['auc']:.5f} | {cv_mean_repr} | {cv_per_fold_repr} "
        f"| {total_seconds:.0f} | {peak_memory_mib:.0f} |\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(row)


def run_submit(config: dict, run_id: str, config_path: Path) -> dict:
    run_dir = _run_dir(config, run_id)
    metrics = _load_metrics(run_dir)
    if "holdout" not in metrics:
        sys.exit(f"{run_dir / 'metrics.json'}: нет стадии holdout — запустите --stage holdout сначала")

    full_iterations = round(metrics["holdout"]["best_iteration"] * config["model"]["full_train_iterations_multiplier"])

    df, X, cat_features, y = _load_train_matrix(config)
    params = _model_params(config, iterations=full_iterations, early_stopping=False)
    model, elapsed = _fit(X, y, cat_features, params, config["model"]["task_type"])

    test_df = build_candidate_features(sort_by_transaction_dt(load_split(config, "test")), config)
    X_test, _ = build_matrix(test_df, config)
    preds = model.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]

    submissions_dir = Path(config["paths"]["submissions_dir"])
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submissions_dir / f"{run_id}.csv"
    pd.DataFrame({"TransactionID": test_df["TransactionID"], "isFraud": preds}).to_csv(submission_path, index=False)

    metrics["submit"] = {
        "full_train_iterations": full_iterations,
        "seconds": elapsed,
        "peak_memory_bytes": _peak_memory_bytes(),
        "submission_path": str(submission_path),
    }
    _save_metrics(run_dir, metrics)
    _append_experiments_log(config, run_id, metrics, config_path)

    print(f"[submit] iterations={full_iterations} time={elapsed:.1f}s -> {submission_path}")
    return metrics


def run_log(config: dict, run_id: str, config_path: Path) -> None:
    """Дописывает строку в experiments.md из уже накопленного metrics.json, не запуская
    обучение — например, после пересчёта только стадии cv для существующего run_id."""
    run_dir = _run_dir(config, run_id)
    metrics = _load_metrics(run_dir)
    missing = [stage for stage in ("holdout", "submit") if stage not in metrics]
    if missing:
        sys.exit(f"{run_dir / 'metrics.json'}: нет стадий {missing} — запустите их перед --stage log")

    _append_experiments_log(config, run_id, metrics, config_path)
    print(f"[log] строка для run_id={run_id} дописана в {config['paths']['experiments_log']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/baseline.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_experiment_config(config_path)
    _check_av_gate(config)

    if args.stage in ("cv", "submit", "log") and args.run_id is None:
        runs_dir = Path(config["paths"]["runs_dir"])
        existing = sorted(p.name for p in runs_dir.glob(f"{config['experiment']}_*")) if runs_dir.exists() else []
        sys.exit(
            f"--run-id обязателен для --stage {args.stage}. "
            f"Доступные run_id для эксперимента {config['experiment']!r}: {existing or '(нет)'}"
        )

    run_id = args.run_id or _run_id(config["experiment"])

    if args.stage in ("holdout", "all"):
        run_holdout(config, run_id)
    if args.stage in ("cv", "all"):
        run_cv(config, run_id)
    if args.stage in ("submit", "all"):
        run_submit(config, run_id, config_path)
    if args.stage == "log":
        run_log(config, run_id, config_path)


if __name__ == "__main__":
    main()
