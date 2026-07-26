"""Слой 1 + AV: строит признаки-кандидаты на train/test и прогоняет adversarial
validation по D-колонкам. Выход: reports/av_d_features.csv, reports/av_d_summary.csv.
"""

from pathlib import Path

from src.adversarial_validation import run_av, save_av_summary, save_av_table, summarize_d_block
from src.config import load_config
from src.data import load_split
from src.families import get_d_columns
from src.features import build_candidate_features

EXPECTED_NORM = {"D1", "D4", "D10", "D11", "D15"}
EXPECTED_UNCLEAR = {"D2"}


def _expected_version(col: str, config: dict) -> str:
    if config["d_features"]["resolution"].get(col) == "excluded":
        return "excluded (решено заранее)"
    if col in EXPECTED_NORM:
        return "norm"
    if col in EXPECTED_UNCLEAR:
        return "unclear (по EDA неопределено)"
    return "raw"


def _actual_version(col: str, av_table) -> str:
    rows = av_table[av_table["column"] == col]
    passing = rows[rows["verdict"] == "pass"]
    if passing.empty:
        return "excluded (обе версии дрейфуют)"
    best = passing.loc[passing["auc"].idxmin()]
    return str(best["version"])


def print_comparison(av_table, config: dict) -> None:
    print("\nСравнение вердиктов AV с ожиданием по EDA (02_feature_eda, §7):")
    header = f"{'колонка':<8}{'ожидание':<28}{'AV-выбор':<28}{'auc_raw':<10}{'auc_norm':<10}{'совпадает?'}"
    print(header)
    print("-" * len(header))

    mismatches = []
    for col in get_d_columns(config):
        expected = _expected_version(col, config)
        actual = _actual_version(col, av_table)
        rows = av_table[av_table["column"] == col]
        auc_raw = rows[rows["version"] == "raw"]["auc"].iloc[0]
        auc_norm = rows[rows["version"] == "norm"]["auc"].iloc[0]

        is_unclear = expected.startswith("unclear")
        is_preset = expected.endswith("(решено заранее)")
        match = "-" if (is_unclear or is_preset) else ("ДА" if expected == actual else "НЕТ")
        if match == "НЕТ":
            mismatches.append(col)

        print(f"{col:<8}{expected:<28}{actual:<28}{auc_raw:<10.4f}{auc_norm:<10.4f}{match}")

    print()
    if mismatches:
        print(f"Расхождения с ожиданием по EDA: {', '.join(mismatches)}")
    else:
        print("Расхождений с ожиданием по EDA (кроме заведомо неясных/решённых заранее) нет.")


def main() -> None:
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Загружаю train...")
    train_df = build_candidate_features(load_split(config, "train"), config)
    print(f"train: {train_df.shape}")

    print("Загружаю test...")
    test_df = build_candidate_features(load_split(config, "test"), config)
    print(f"test: {test_df.shape}")

    d_columns = get_d_columns(config)

    print("Запускаю AV по D-колонкам...")
    av_table = run_av(train_df, test_df, d_columns, config)
    save_av_table(av_table, reports_dir / "av_d_features.csv")
    print(f"Сохранено: {reports_dir / 'av_d_features.csv'}")

    print("Считаю суммарную различимость train/test по D-блоку до/после отбора...")
    summary = summarize_d_block(train_df, test_df, d_columns, av_table, config)
    save_av_summary(summary, reports_dir / "av_d_summary.csv")
    print(f"Сохранено: {reports_dir / 'av_d_summary.csv'}")
    print(
        f"joint AUC до отбора: {summary['auc_before']:.4f} ({summary['n_columns_before']} колонок), "
        f"после отбора: {summary['auc_after']:.4f} ({summary['n_columns_after']} колонок)"
    )

    print_comparison(av_table, config)


if __name__ == "__main__":
    main()
