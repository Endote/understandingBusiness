#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def summary(values: np.ndarray, label: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "series": label,
        "rows": int(len(values)),
        "nonzero_rate": float((values > 0).mean()),
        "unit_sum": float(values.sum()),
        "mean_all": float(values.mean()),
        "mean_nonzero": float(values[values > 0].mean()) if (values > 0).any() else 0.0,
        "p50_all": float(np.quantile(values, 0.50)),
        "p90_all": float(np.quantile(values, 0.90)),
        "p95_all": float(np.quantile(values, 0.95)),
        "p99_all": float(np.quantile(values, 0.99)),
        "top10_share": top_share(values, 0.10),
        "top5_share": top_share(values, 0.05),
        "top1_share": top_share(values, 0.01),
    }


def top_share(values: np.ndarray, frac: float) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    count = max(1, int(round(len(values) * frac)))
    return float(np.sort(values)[::-1][:count].sum() / total)


def deciles(values: np.ndarray, label: str) -> pd.DataFrame:
    work = pd.DataFrame({"value": np.asarray(values, dtype=np.float64)})
    work["decile_num"] = pd.qcut(work["value"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    total = float(work["value"].sum())
    rows = []
    for decile_num, group in work.groupby("decile_num", observed=True):
        vals = group["value"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "series": label,
                "decile": f"D{int(decile_num)}",
                "rows": int(len(vals)),
                "unit_sum": float(vals.sum()),
                "unit_share": float(vals.sum() / total) if total > 0 else 0.0,
                "mean_units": float(vals.mean()),
                "min_units": float(vals.min()),
                "median_units": float(np.quantile(vals, 0.50)),
                "max_units": float(vals.max()),
                "zero_rate": float((vals <= 0).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def actual_series_from_summary(summary_path: Path, dataset: str, label: str) -> dict[str, object]:
    df = pd.read_csv(summary_path)
    row = df[(df["dataset"] == dataset) & (df["scope"] == "all_rows")].iloc[0]
    return {
        "series": label,
        "rows": int(row["rows"]),
        "nonzero_rate": float(row["nonzero_rate"]),
        "unit_sum": float(row["unit_sum"]),
        "mean_all": float(row["mean_units_all_rows"]),
        "mean_nonzero": float(row["mean_units_nonzero_rows"]),
        "p50_all": float(row["p50_all_rows"]),
        "p90_all": float(row["p90_all_rows"]),
        "p95_all": float(row["p95_all_rows"]),
        "p99_all": float(row["p99_all_rows"]),
        "top10_share": np.nan,
        "top5_share": np.nan,
        "top1_share": np.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final July prediction comparison tables.")
    parser.add_argument("--old-scoring-dir", type=Path, required=True)
    parser.add_argument("--new-scoring-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.new_scoring_dir / "final_comparison_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_summary = args.old_scoring_dir / "decile_compare" / "unit_distribution_summary.csv"
    new_summary = args.new_scoring_dir / "decile_compare" / "unit_distribution_summary.csv"
    old_pred = pd.read_parquet(args.old_scoring_dir / "july_holdout_weeklies_predictions.parquet", columns=["july_holdout_prediction"])
    new_pred = pd.read_parquet(args.new_scoring_dir / "july_holdout_weeklies_predictions.parquet", columns=["july_holdout_prediction", "positive_regressor_prediction"])
    twotier_path = args.new_scoring_dir / "twotier_gate_grid" / "july_holdout_weeklies_predictions_twotier_best.parquet"
    twotier = pd.read_parquet(twotier_path, columns=["july_holdout_prediction_twotier_best"])

    rows = [
        actual_series_from_summary(new_summary, "full_labeled_actual", "Full labeled actual"),
        actual_series_from_summary(new_summary, "model_train_split_actual", "Model train split actual"),
        summary(old_pred["july_holdout_prediction"].to_numpy(dtype=np.float64), "Old July hard gate"),
        summary(new_pred["july_holdout_prediction"].to_numpy(dtype=np.float64), "Full-label naive soft floor"),
        summary(twotier["july_holdout_prediction_twotier_best"].to_numpy(dtype=np.float64), "Full-label two-tier soft gate"),
        summary(new_pred["positive_regressor_prediction"].to_numpy(dtype=np.float64), "Full-label pre-gate regressor"),
    ]
    summary_frame = pd.DataFrame.from_records(rows)
    summary_frame.to_csv(output_dir / "summary_comparison.csv", index=False)

    decile_frame = pd.concat(
        [
            pd.read_csv(args.old_scoring_dir / "decile_compare" / "unit_decile_comparison.csv").query(
                "dataset in ['full_labeled_actual', 'model_train_split_actual'] and scope == 'all_rows'"
            ).rename(columns={"dataset": "series"})[
                ["series", "decile", "rows", "unit_sum", "unit_share", "mean_units", "min_units", "median_units", "max_units", "zero_row_frac"]
            ],
            deciles(old_pred["july_holdout_prediction"].to_numpy(dtype=np.float64), "Old July hard gate").rename(columns={"zero_rate": "zero_row_frac"}),
            deciles(new_pred["july_holdout_prediction"].to_numpy(dtype=np.float64), "Full-label naive soft floor").rename(columns={"zero_rate": "zero_row_frac"}),
            deciles(twotier["july_holdout_prediction_twotier_best"].to_numpy(dtype=np.float64), "Full-label two-tier soft gate").rename(columns={"zero_rate": "zero_row_frac"}),
        ],
        ignore_index=True,
    )
    decile_frame.to_csv(output_dir / "all_row_decile_comparison.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "old_scoring_dir": str(args.old_scoring_dir),
                "new_scoring_dir": str(args.new_scoring_dir),
                "twotier_prediction_path": str(twotier_path),
                "outputs": {
                    "summary_comparison": str(output_dir / "summary_comparison.csv"),
                    "all_row_decile_comparison": str(output_dir / "all_row_decile_comparison.csv"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
