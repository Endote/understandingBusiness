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


def run_psql_csv(db_uri: str, query: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="paperrush_group_align_") as tmp_dir:
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
    where dp.type = 'Weeklies'
      and fs.drawqty <> 0
    {group_by}
    """


def actual_projection_query() -> str:
    return """
    select
        coalesce(nullif(trim(dp.title), ''), 'UNKNOWN') as title,
        coalesce(nullif(trim(dp.segment), ''), 'UNKNOWN') as segment,
        coalesce(nullif(trim(dp.subsegment), ''), 'UNKNOWN') as subsegment,
        coalesce(nullif(trim(ds.classoftrade), ''), 'UNKNOWN') as classoftrade,
        coalesce(nullif(trim(ds.store_chain), ''), 'UNKNOWN') as store_chain,
        greatest(fs.soldqty, 0)::double precision as actual_units
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    join core.dim_store ds on ds.store_id = fs.store_id
    where dp.type = 'Weeklies'
      and fs.drawqty <> 0
    """


def group_prediction(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            predicted_rows=("ceiled_prediction", "size"),
            predicted_nonzero_rows=("ceiled_prediction", lambda values: int((values > 0).sum())),
            predicted_units=("ceiled_prediction", "sum"),
            predicted_mean_all=("ceiled_prediction", "mean"),
        )
        .reset_index()
    )
    grouped["predicted_mean_nonzero"] = (
        frame.loc[frame["ceiled_prediction"] > 0]
        .groupby(group_cols, dropna=False, observed=True)["ceiled_prediction"]
        .mean()
        .reindex(pd.MultiIndex.from_frame(grouped[group_cols]) if len(group_cols) > 1 else grouped[group_cols[0]])
        .to_numpy()
    )
    grouped["predicted_mean_nonzero"] = grouped["predicted_mean_nonzero"].fillna(0.0)
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


def metric_row(name: str, table: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    unit_diff = (table["predicted_unit_share"] - table["actual_unit_share"]).abs()
    row_diff = (table["predicted_row_share"] - table["actual_row_share"]).abs()
    max_idx = int(unit_diff.idxmax())
    group_label = " | ".join(str(table.loc[max_idx, col]) for col in group_cols)
    return {
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
    parser = argparse.ArgumentParser(description="Compare full labeled actual unit distribution against ceiled July two-tier predictions.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_path = args.scoring_dir / "twotier_gate_grid" / "july_holdout_weeklies_predictions_twotier_best.parquet"
    output_dir = args.scoring_dir / "ceiled_group_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(pred_path)
    predictions["ceiled_prediction"] = np.ceil(predictions["july_holdout_prediction_twotier_best"].to_numpy(dtype=np.float64))

    metrics = []
    for name, cols in GROUPS.items():
        actual = run_psql_csv(args.db_uri, actual_query(cols))
        predicted = group_prediction(predictions, cols)
        table = compare_tables(actual, predicted, cols)
        table.to_csv(output_dir / f"alignment_by_{name}.csv", index=False)
        metrics.append(metric_row(name, table, cols))

    metrics_frame = pd.DataFrame.from_records(metrics).sort_values("unit_share_tvd")
    metrics_frame.to_csv(output_dir / "alignment_metrics.csv", index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scoring_dir": str(args.scoring_dir),
        "prediction_path": str(pred_path),
        "rounding": "ceil(row prediction)",
        "outputs": {
            "alignment_metrics": str(output_dir / "alignment_metrics.csv"),
            **{name: str(output_dir / f"alignment_by_{name}.csv") for name in GROUPS},
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
