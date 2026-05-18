#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_sip_ceiled_group_alignment import (
    DEFAULT_DB_URI,
    GROUPS,
    actual_query,
    actual_units_query,
    compare_tables,
    decile_table,
    group_prediction,
    metric_row,
    run_psql_csv,
    series_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full labeled SIP actual distribution against latest SIP July prediction.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--prediction-column", default="july_holdout_prediction")
    parser.add_argument("--ceil-prediction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_path = args.scoring_dir / "july_holdout_sip_predictions.parquet"
    output_dir = args.scoring_dir / ("latest_prediction_group_alignment_ceil" if args.ceil_prediction else "latest_prediction_group_alignment")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(prediction_path)
    if args.prediction_column not in predictions.columns:
        print(f"missing prediction column: {args.prediction_column}", file=sys.stderr)
        return 2

    actual_units = run_psql_csv(args.db_uri, actual_units_query())["actual_units"].to_numpy(dtype="float64")
    if args.ceil_prediction:
        predictions[args.prediction_column] = np.ceil(predictions[args.prediction_column].to_numpy(dtype="float64"))
    prediction_values = predictions[args.prediction_column].to_numpy(dtype="float64")

    actual_label = "full_labeled_actual"
    prediction_label = args.prediction_column
    summary_frame = pd.DataFrame.from_records(
        [
            series_summary(actual_units, actual_label),
            series_summary(prediction_values, prediction_label),
        ]
    )
    summary_frame.to_csv(output_dir / "summary_comparison.csv", index=False)

    decile_frame = pd.concat(
        [
            decile_table(actual_units, actual_label, "actual_unchanged"),
            decile_table(prediction_values, prediction_label, "raw_prediction"),
        ],
        ignore_index=True,
    ).sort_values(["series", "decile_num"])
    decile_frame.to_csv(output_dir / "decile_comparison.csv", index=False)
    mean_wide = decile_frame.pivot(index="decile", columns="series", values="mean_units").loc[[f"D{i}" for i in range(1, 11)]]
    median_wide = decile_frame.pivot(index="decile", columns="series", values="median_units").loc[[f"D{i}" for i in range(1, 11)]]
    mean_wide.to_csv(output_dir / "decile_mean_units_wide.csv")
    median_wide.to_csv(output_dir / "decile_median_units_wide.csv")

    metrics = []
    for name, cols in GROUPS.items():
        actual = run_psql_csv(args.db_uri, actual_query(cols))
        predicted = group_prediction(predictions, cols, args.prediction_column)
        table = compare_tables(actual, predicted, cols)
        table.to_csv(output_dir / f"alignment_by_{name}_{prediction_label}.csv", index=False)
        metrics.append(metric_row(prediction_label, name, table, cols))
    metrics_frame = pd.DataFrame.from_records(metrics).sort_values("unit_share_tvd")
    metrics_frame.to_csv(output_dir / "alignment_metrics.csv", index=False)

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scoring_dir": str(args.scoring_dir),
        "prediction_path": str(prediction_path),
        "prediction_column": args.prediction_column,
        "ceil_prediction": bool(args.ceil_prediction),
        "outputs": {
            "summary_comparison": str(output_dir / "summary_comparison.csv"),
            "alignment_metrics": str(output_dir / "alignment_metrics.csv"),
            "decile_comparison": str(output_dir / "decile_comparison.csv"),
            "decile_mean_units_wide": str(output_dir / "decile_mean_units_wide.csv"),
            "decile_median_units_wide": str(output_dir / "decile_median_units_wide.csv"),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(summary_frame.to_string(index=False), flush=True)
    print("\nALIGNMENT METRICS", flush=True)
    print(metrics_frame.to_string(index=False), flush=True)
    print("\nMEAN DECILES", flush=True)
    print(mean_wide.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)
    print("\nMEDIAN DECILES", flush=True)
    print(median_wide.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
