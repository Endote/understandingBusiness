#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def decile_table(values: np.ndarray, label: str) -> pd.DataFrame:
    work = pd.DataFrame({"value": np.ceil(np.asarray(values, dtype=np.float64))})
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
                "zero_row_frac": float((vals <= 0).mean()),
                "rounded_operation": "ceil_predictions",
            }
        )
    return pd.DataFrame.from_records(rows)


def prediction_summary(values: np.ndarray, label: str) -> dict[str, object]:
    values = np.ceil(np.asarray(values, dtype=np.float64))
    nonzero = values[values > 0]
    return {
        "series": label,
        "rows": int(len(values)),
        "nonzero_rate": float((values > 0).mean()),
        "unit_sum": float(values.sum()),
        "mean_all": float(values.mean()),
        "mean_nonzero": float(nonzero.mean()) if len(nonzero) else 0.0,
        "p50_all": float(np.quantile(values, 0.50)),
        "p90_all": float(np.quantile(values, 0.90)),
        "p95_all": float(np.quantile(values, 0.95)),
        "p99_all": float(np.quantile(values, 0.99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round July predictions up and rebuild decile mean/median comparisons.")
    parser.add_argument("--old-scoring-dir", type=Path, required=True)
    parser.add_argument("--new-scoring-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.new_scoring_dir / "final_comparison_report"
    base = pd.read_csv(report_dir / "all_row_decile_comparison.csv")
    actual = base.loc[base["series"].isin(["full_labeled_actual", "model_train_split_actual"])].copy()
    actual["rounded_operation"] = "actual_unchanged"

    old = pd.read_parquet(args.old_scoring_dir / "july_holdout_weeklies_predictions.parquet", columns=["july_holdout_prediction"])
    new = pd.read_parquet(
        args.new_scoring_dir / "twotier_gate_grid" / "july_holdout_weeklies_predictions_twotier_best.parquet",
        columns=["july_holdout_prediction_twotier_best"],
    )
    rounded = pd.concat(
        [
            actual,
            decile_table(old["july_holdout_prediction"].to_numpy(dtype=np.float64), "Old July hard gate ceil"),
            decile_table(
                new["july_holdout_prediction_twotier_best"].to_numpy(dtype=np.float64),
                "Full-label two-tier soft gate ceil",
            ),
        ],
        ignore_index=True,
    )
    rounded["decile_num"] = rounded["decile"].str.replace("D", "", regex=False).astype(int)
    rounded = rounded.sort_values(["series", "decile_num"])
    rounded.to_csv(report_dir / "all_row_decile_comparison_predictions_ceiled.csv", index=False)

    series_order = [
        "full_labeled_actual",
        "model_train_split_actual",
        "Old July hard gate ceil",
        "Full-label two-tier soft gate ceil",
    ]
    wide_mean = rounded.pivot(index="decile", columns="series", values="mean_units").loc[
        [f"D{i}" for i in range(1, 11)], series_order
    ]
    wide_median = rounded.pivot(index="decile", columns="series", values="median_units").loc[
        [f"D{i}" for i in range(1, 11)], series_order
    ]
    wide_mean.to_csv(report_dir / "decile_mean_units_predictions_ceiled_wide.csv")
    wide_median.to_csv(report_dir / "decile_median_units_predictions_ceiled_wide.csv")
    pd.DataFrame.from_records(
        [
            prediction_summary(old["july_holdout_prediction"].to_numpy(dtype=np.float64), "Old July hard gate ceil"),
            prediction_summary(
                new["july_holdout_prediction_twotier_best"].to_numpy(dtype=np.float64),
                "Full-label two-tier soft gate ceil",
            ),
        ]
    ).to_csv(report_dir / "summary_predictions_ceiled.csv", index=False)

    print("MEANS")
    print(wide_mean.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nMEDIANS")
    print(wide_median.to_string(float_format=lambda value: f"{value:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
