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


GROUPS: dict[str, list[str]] = {
    "title": ["title"],
    "segment": ["segment"],
    "subsegment": ["subsegment"],
    "classoftrade": ["classoftrade"],
    "store_chain": ["store_chain"],
    "title_segment": ["title", "segment"],
    "title_classoftrade": ["title", "classoftrade"],
    "segment_classoftrade": ["segment", "classoftrade"],
}
PREDICTION_COLUMNS = {
    "sip_892_ceil": "sip_prediction_892",
    "sip_lift_11655_ceil": "sip_prediction_lift_11655",
}


def run_psql_csv(db_uri: str, query: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="paperrush_sip_align_") as tmp_dir:
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


def actual_query(group_cols: list[str]) -> str:
    select_cols = ", ".join(group_cols)
    group_by = ", ".join(group_cols)
    if select_cols:
        select_cols = f"{select_cols},"
        group_by = f"group by {group_by}"
    return f"""
    select
        {select_cols}
        count(*)::bigint as actual_rows,
        count(*) filter (where fs.soldqty > 0)::bigint as actual_nonzero_rows,
        coalesce(sum(greatest(fs.soldqty, 0)), 0)::double precision as actual_units,
        coalesce(avg(greatest(fs.soldqty, 0)), 0)::double precision as actual_mean_all,
        coalesce(avg(greatest(fs.soldqty, 0)) filter (where fs.soldqty > 0), 0)::double precision as actual_mean_nonzero
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    join core.dim_store ds on ds.store_id = fs.store_id
    where dp.type = 'SIP'
      and fs.drawqty <> 0
    {group_by}
    """


def actual_units_query() -> str:
    return """
    select greatest(fs.soldqty, 0)::double precision as actual_units
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    where dp.type = 'SIP'
      and fs.drawqty <> 0
    """


def series_summary(values: np.ndarray, label: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
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


def decile_table(values: np.ndarray, label: str, rounded_operation: str) -> pd.DataFrame:
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
                "decile_num": int(decile_num),
                "rows": int(len(vals)),
                "unit_sum": float(vals.sum()),
                "unit_share": float(vals.sum() / total) if total > 0 else 0.0,
                "mean_units": float(vals.mean()),
                "min_units": float(vals.min()),
                "median_units": float(np.quantile(vals, 0.50)),
                "max_units": float(vals.max()),
                "zero_row_frac": float((vals <= 0).mean()),
                "rounded_operation": rounded_operation,
            }
        )
    return pd.DataFrame.from_records(rows)


def group_prediction(frame: pd.DataFrame, group_cols: list[str], prediction_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            predicted_rows=(prediction_col, "size"),
            predicted_nonzero_rows=(prediction_col, lambda values: int((values > 0).sum())),
            predicted_units=(prediction_col, "sum"),
            predicted_mean_all=(prediction_col, "mean"),
        )
        .reset_index()
    )
    nonzero = frame.loc[frame[prediction_col] > 0]
    if nonzero.empty:
        grouped["predicted_mean_nonzero"] = 0.0
        return grouped
    nonzero_mean = nonzero.groupby(group_cols, dropna=False, observed=True)[prediction_col].mean()
    if len(group_cols) > 1:
        grouped["predicted_mean_nonzero"] = pd.MultiIndex.from_frame(grouped[group_cols]).map(nonzero_mean).fillna(0.0).to_numpy()
    else:
        grouped["predicted_mean_nonzero"] = grouped[group_cols[0]].map(nonzero_mean).fillna(0.0).to_numpy()
    return grouped


def compare_tables(actual: pd.DataFrame, predicted: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = actual.merge(predicted, how="outer", on=group_cols)
    for col in [
        "actual_rows",
        "actual_nonzero_rows",
        "actual_units",
        "actual_mean_all",
        "actual_mean_nonzero",
        "predicted_rows",
        "predicted_nonzero_rows",
        "predicted_units",
        "predicted_mean_all",
        "predicted_mean_nonzero",
    ]:
        table[col] = table[col].fillna(0.0)
    actual_rows_total = float(table["actual_rows"].sum())
    predicted_rows_total = float(table["predicted_rows"].sum())
    actual_units_total = float(table["actual_units"].sum())
    predicted_units_total = float(table["predicted_units"].sum())
    table["actual_row_share"] = table["actual_rows"] / actual_rows_total
    table["predicted_row_share"] = table["predicted_rows"] / predicted_rows_total
    table["row_share_delta_pp"] = 100.0 * (table["predicted_row_share"] - table["actual_row_share"])
    table["actual_unit_share"] = table["actual_units"] / actual_units_total
    table["predicted_unit_share"] = table["predicted_units"] / predicted_units_total
    table["unit_share_delta_pp"] = 100.0 * (table["predicted_unit_share"] - table["actual_unit_share"])
    table["predicted_to_actual_unit_share_ratio"] = np.divide(
        table["predicted_unit_share"],
        table["actual_unit_share"],
        out=np.full(len(table), np.nan),
        where=table["actual_unit_share"].to_numpy(dtype=float) > 0,
    )
    table["actual_nonzero_rate"] = np.divide(
        table["actual_nonzero_rows"],
        table["actual_rows"],
        out=np.zeros(len(table), dtype=float),
        where=table["actual_rows"].to_numpy(dtype=float) > 0,
    )
    table["predicted_nonzero_rate"] = np.divide(
        table["predicted_nonzero_rows"],
        table["predicted_rows"],
        out=np.zeros(len(table), dtype=float),
        where=table["predicted_rows"].to_numpy(dtype=float) > 0,
    )
    return table.sort_values("predicted_units", ascending=False)


def metric_row(series: str, name: str, table: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    unit_diff = (table["predicted_unit_share"] - table["actual_unit_share"]).abs()
    row_diff = (table["predicted_row_share"] - table["actual_row_share"]).abs()
    max_idx = int(unit_diff.idxmax())
    group_label = " | ".join(str(table.loc[max_idx, col]) for col in group_cols)
    return {
        "series": series,
        "grouping": name,
        "groups": int(len(table)),
        "unit_share_tvd": float(0.5 * unit_diff.sum()),
        "unit_share_l1": float(unit_diff.sum()),
        "row_share_tvd": float(0.5 * row_diff.sum()),
        "row_share_l1": float(row_diff.sum()),
        "max_abs_unit_shift_pp": float(100.0 * unit_diff.max()),
        "max_unit_shift_group": group_label,
        "predicted_units_total": float(table["predicted_units"].sum()),
        "actual_units_total": float(table["actual_units"].sum()),
        "predicted_rows_total": int(table["predicted_rows"].sum()),
        "actual_rows_total": int(table["actual_rows"].sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full labeled SIP actual distribution against ceiled SIP predictions.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.scoring_dir / "ceiled_group_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(args.scoring_dir / "july_holdout_sip_predictions.parquet")
    actual_units = run_psql_csv(args.db_uri, actual_units_query())["actual_units"].to_numpy(dtype=np.float64)

    summary_rows = [series_summary(actual_units, "full_labeled_actual")]
    decile_frames = [decile_table(actual_units, "full_labeled_actual", "actual_unchanged")]
    metrics = []
    for series, source_col in PREDICTION_COLUMNS.items():
        pred_col = f"{series}_value"
        predictions[pred_col] = np.ceil(predictions[source_col].to_numpy(dtype=np.float64))
        values = predictions[pred_col].to_numpy(dtype=np.float64)
        summary_rows.append(series_summary(values, series))
        decile_frames.append(decile_table(values, series, "ceil_predictions"))
        for name, cols in GROUPS.items():
            actual = run_psql_csv(args.db_uri, actual_query(cols))
            predicted = group_prediction(predictions, cols, pred_col)
            table = compare_tables(actual, predicted, cols)
            table.to_csv(output_dir / f"alignment_by_{name}_{series}.csv", index=False)
            metrics.append(metric_row(series, name, table, cols))

    summary_frame = pd.DataFrame.from_records(summary_rows)
    summary_frame.to_csv(output_dir / "summary_comparison.csv", index=False)
    decile_frame = pd.concat(decile_frames, ignore_index=True).sort_values(["series", "decile_num"])
    decile_frame.to_csv(output_dir / "decile_comparison_ceiled.csv", index=False)
    mean_wide = decile_frame.pivot(index="decile", columns="series", values="mean_units").loc[[f"D{i}" for i in range(1, 11)]]
    median_wide = decile_frame.pivot(index="decile", columns="series", values="median_units").loc[[f"D{i}" for i in range(1, 11)]]
    mean_wide.to_csv(output_dir / "decile_mean_units_ceiled_wide.csv")
    median_wide.to_csv(output_dir / "decile_median_units_ceiled_wide.csv")
    metrics_frame = pd.DataFrame.from_records(metrics).sort_values(["series", "unit_share_tvd"])
    metrics_frame.to_csv(output_dir / "alignment_metrics.csv", index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scoring_dir": str(args.scoring_dir),
        "rounding": "ceil(row prediction)",
        "outputs": {
            "summary_comparison": str(output_dir / "summary_comparison.csv"),
            "alignment_metrics": str(output_dir / "alignment_metrics.csv"),
            "decile_comparison_ceiled": str(output_dir / "decile_comparison_ceiled.csv"),
            "decile_mean_units_ceiled_wide": str(output_dir / "decile_mean_units_ceiled_wide.csv"),
            "decile_median_units_ceiled_wide": str(output_dir / "decile_median_units_ceiled_wide.csv"),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False), flush=True)
    print("\nALIGNMENT METRICS")
    print(metrics_frame.to_string(index=False), flush=True)
    print("\nMEAN DECILES")
    print(mean_wide.to_string(float_format=lambda value: f"{value:.3f}"), flush=True)
    print("\nMEDIAN DECILES")
    print(median_wide.to_string(float_format=lambda value: f"{value:.3f}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
