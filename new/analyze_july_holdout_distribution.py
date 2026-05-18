#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_modeling_datasets import DEFAULT_DB_URI


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = NEW_DIR / "output" / "JulyScoring"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_psql_csv(db_uri: str, query: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="paperrush_distribution_") as tmp_dir:
        csv_path = Path(tmp_dir) / "query.csv"
        copy_cmd = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
        with csv_path.open("wb") as handle:
            result = subprocess.run(
                ["psql", db_uri, "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-c", copy_cmd],
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return pd.read_csv(csv_path)


def latest_scoring_dir(output_root: Path, family: str) -> Path:
    family_dir = output_root / family.lower()
    runs = sorted((path for path in family_dir.iterdir() if path.is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No scoring runs found under {family_dir}")
    return runs[0]


def labeled_group_query(group_cols: list[str], where_extra: str = "") -> str:
    select_cols = ", ".join(group_cols)
    group_by = ", ".join(group_cols)
    if select_cols:
        select_cols = f"{select_cols},"
        group_by = f"group by {group_by}"
    return f"""
    select
        {select_cols}
        count(*)::bigint as labeled_rows,
        count(*) filter (where fs.soldqty > 0)::bigint as positive_rows,
        count(*) filter (where fs.soldqty = 0)::bigint as zero_rows,
        count(*) filter (where fs.soldqty < 0)::bigint as negative_rows,
        coalesce(sum(greatest(fs.soldqty, 0)), 0)::double precision as positive_sales_units,
        coalesce(avg(greatest(fs.soldqty, 0)) filter (where fs.soldqty > 0), 0)::double precision as avg_positive_units
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    where dp.type = 'Weeklies'
      and fs.drawqty <> 0
      {where_extra}
    {group_by}
    """


def holdout_group_query(group_cols: list[str]) -> str:
    select_cols = ", ".join(group_cols)
    group_by = ", ".join(group_cols)
    if select_cols:
        select_cols = f"{select_cols},"
        group_by = f"group by {group_by}"
    return f"""
    select
        {select_cols}
        count(*)::bigint as holdout_rows
    from holdout.printing_schedule ps
    join holdout.dim_product dp on dp.product_id = ps.product_id
    where dp.type = 'Weeklies'
    {group_by}
    """


def score_group_predictions(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        predictions.groupby(group_cols, dropna=False, observed=True)
        .agg(
            scored_rows=("july_holdout_prediction", "size"),
            predicted_units=("july_holdout_prediction", "sum"),
            prediction_mean=("july_holdout_prediction", "mean"),
            raw_probability_mean=("incidence_raw_probability", "mean"),
            calibrated_probability_mean=("incidence_calibrated_probability", "mean"),
            gated_rows=("incidence_gate", "sum"),
            positive_regressor_mean=("positive_regressor_prediction", "mean"),
        )
        .reset_index()
    )


def compare_group(
    labeled: pd.DataFrame,
    holdout: pd.DataFrame,
    scored: pd.DataFrame,
    group_cols: list[str],
    labeled_total: int,
    holdout_total: int,
) -> pd.DataFrame:
    key = group_cols
    merged = labeled.merge(holdout, how="outer", on=key).merge(scored, how="outer", on=key)
    count_cols = [
        "labeled_rows",
        "positive_rows",
        "zero_rows",
        "negative_rows",
        "holdout_rows",
        "scored_rows",
        "gated_rows",
    ]
    for col in count_cols:
        if col in merged:
            merged[col] = merged[col].fillna(0)
    for col in ["positive_sales_units", "avg_positive_units", "predicted_units", "prediction_mean", "raw_probability_mean", "calibrated_probability_mean", "positive_regressor_mean"]:
        if col in merged:
            merged[col] = merged[col].fillna(0.0)
    merged["labeled_row_frac"] = merged["labeled_rows"].astype(float) / labeled_total
    merged["holdout_row_frac"] = merged["holdout_rows"].astype(float) / holdout_total
    merged["holdout_minus_labeled_frac_pp"] = 100.0 * (merged["holdout_row_frac"] - merged["labeled_row_frac"])
    merged["holdout_to_labeled_frac_ratio"] = np.divide(
        merged["holdout_row_frac"],
        merged["labeled_row_frac"],
        out=np.full(len(merged), np.nan),
        where=merged["labeled_row_frac"].to_numpy(dtype=float) > 0,
    )
    merged["historical_positive_rate"] = np.divide(
        merged["positive_rows"].astype(float),
        merged["labeled_rows"].astype(float),
        out=np.zeros(len(merged), dtype=float),
        where=merged["labeled_rows"].to_numpy(dtype=float) > 0,
    )
    merged["historical_zero_rate"] = np.divide(
        merged["zero_rows"].astype(float),
        merged["labeled_rows"].astype(float),
        out=np.zeros(len(merged), dtype=float),
        where=merged["labeled_rows"].to_numpy(dtype=float) > 0,
    )
    merged["historical_negative_rate"] = np.divide(
        merged["negative_rows"].astype(float),
        merged["labeled_rows"].astype(float),
        out=np.zeros(len(merged), dtype=float),
        where=merged["labeled_rows"].to_numpy(dtype=float) > 0,
    )
    merged["scoring_gate_rate"] = np.divide(
        merged["gated_rows"].astype(float),
        merged["scored_rows"].astype(float),
        out=np.zeros(len(merged), dtype=float),
        where=merged["scored_rows"].to_numpy(dtype=float) > 0,
    )
    merged["calibrated_minus_historical_positive_rate"] = merged["calibrated_probability_mean"] - merged["historical_positive_rate"]
    merged["gate_minus_historical_positive_rate"] = merged["scoring_gate_rate"] - merged["historical_positive_rate"]
    return merged.sort_values("holdout_rows", ascending=False)


def distribution_metric(table: pd.DataFrame, label: str) -> dict[str, object]:
    diff = table["holdout_row_frac"].fillna(0.0) - table["labeled_row_frac"].fillna(0.0)
    abs_diff = diff.abs()
    max_idx = int(abs_diff.idxmax())
    return {
        "dimension": label,
        "groups": int(len(table)),
        "total_variation_distance": float(0.5 * abs_diff.sum()),
        "l1_abs_fraction_shift": float(abs_diff.sum()),
        "max_abs_shift_pp": float(100.0 * abs_diff.max()),
        "max_shift_group": " | ".join(str(table.loc[max_idx, col]) for col in table.columns[: table.columns.get_loc("labeled_rows")]),
        "max_shift_holdout_frac": float(table.loc[max_idx, "holdout_row_frac"]),
        "max_shift_labeled_frac": float(table.loc[max_idx, "labeled_row_frac"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare July holdout distribution to labeled Weeklies history.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--scoring-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    scoring_dir = args.scoring_dir or latest_scoring_dir(args.output_root, "weeklies")
    output_dir = scoring_dir / "distribution_compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = scoring_dir / "july_holdout_weeklies_predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    log("Loading scored July holdout predictions")
    predictions = pd.read_parquet(predictions_path)
    log("Querying current database distribution aggregates")
    overall_labeled = run_psql_csv(args.db_uri, labeled_group_query([]))
    overall_holdout = run_psql_csv(args.db_uri, holdout_group_query([]))
    labeled_total = int(overall_labeled["labeled_rows"].iloc[0])
    holdout_total = int(overall_holdout["holdout_rows"].iloc[0])

    dimensions = {
        "title": ["title"],
        "segment": ["segment"],
        "title_segment": ["title", "segment"],
    }
    metric_rows = []
    output_paths = {}
    for name, cols in dimensions.items():
        log(f"Comparing {name}")
        labeled = run_psql_csv(args.db_uri, labeled_group_query(cols))
        holdout = run_psql_csv(args.db_uri, holdout_group_query(cols))
        scored = score_group_predictions(predictions, cols)
        table = compare_group(labeled, holdout, scored, cols, labeled_total, holdout_total)
        path = output_dir / f"compare_by_{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = str(path)
        metric_rows.append(distribution_metric(table, name))

    overall = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scoring_dir": str(scoring_dir),
        "labeled_rows": labeled_total,
        "holdout_rows": holdout_total,
        "labeled_positive_rows": int(overall_labeled["positive_rows"].iloc[0]),
        "labeled_zero_rows": int(overall_labeled["zero_rows"].iloc[0]),
        "labeled_negative_rows": int(overall_labeled["negative_rows"].iloc[0]),
        "labeled_positive_rate": float(overall_labeled["positive_rows"].iloc[0] / labeled_total),
        "labeled_zero_rate": float(overall_labeled["zero_rows"].iloc[0] / labeled_total),
        "labeled_negative_rate": float(overall_labeled["negative_rows"].iloc[0] / labeled_total),
        "holdout_actual_class_rates": "unavailable: holdout.printing_schedule has store_id/product_id only, no soldqty label",
        "scored_raw_probability_mean": float(predictions["incidence_raw_probability"].mean()),
        "scored_calibrated_probability_mean": float(predictions["incidence_calibrated_probability"].mean()),
        "scored_gate_rate": float(predictions["incidence_gate"].mean()),
        "scored_positive_regressor_sum": float(predictions["positive_regressor_prediction"].sum()),
        "scored_final_prediction_sum": float(predictions["july_holdout_prediction"].sum()),
        "distribution_metrics": metric_rows,
        "output_paths": output_paths,
    }
    (output_dir / "distribution_compare_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    pd.DataFrame.from_records(metric_rows).to_csv(output_dir / "distribution_shift_metrics.csv", index=False)
    log(f"Wrote distribution comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
