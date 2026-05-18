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
    with tempfile.TemporaryDirectory(prefix="paperrush_deciles_") as tmp_dir:
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


def latest_scoring_dir(output_root: Path) -> Path:
    family_dir = output_root / "weeklies"
    runs = sorted((path for path in family_dir.iterdir() if path.is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No scoring runs found under {family_dir}")
    return runs[0]


def labeled_query() -> str:
    return """
    select
        greatest(fs.soldqty, 0)::double precision as sold_units,
        fs.soldqty::double precision as raw_soldqty,
        dp.onsaledate,
        coalesce(nullif(trim(dp.title), ''), 'UNKNOWN') as title,
        coalesce(nullif(trim(dp.segment), ''), 'UNKNOWN') as segment
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    where dp.type = 'Weeklies'
      and fs.drawqty <> 0
    """


def decile_table(values: np.ndarray, dataset: str, scope: str, ascending: bool = True) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    work = pd.DataFrame({"value": values})
    if len(work) == 0:
        return pd.DataFrame()
    if ascending:
        ranks = work["value"].rank(method="first", ascending=True)
    else:
        ranks = work["value"].rank(method="first", ascending=False)
    work["decile_num"] = pd.qcut(ranks, q=min(10, len(work)), labels=False, duplicates="drop") + 1
    total_units = float(work["value"].sum())
    rows = []
    for decile_num, group in work.groupby("decile_num", observed=True):
        group_values = group["value"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "scope": scope,
                "decile": f"D{int(decile_num)}",
                "decile_num": int(decile_num),
                "rows": int(len(group)),
                "row_frac": float(len(group) / len(work)),
                "unit_sum": float(group_values.sum()),
                "unit_share": float(group_values.sum() / total_units) if total_units > 0 else 0.0,
                "mean_units": float(group_values.mean()),
                "min_units": float(group_values.min()),
                "p25_units": float(np.quantile(group_values, 0.25)),
                "median_units": float(np.quantile(group_values, 0.50)),
                "p75_units": float(np.quantile(group_values, 0.75)),
                "max_units": float(group_values.max()),
                "zero_rows": int((group_values <= 0).sum()),
                "zero_row_frac": float((group_values <= 0).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def top_concentration(values: np.ndarray, dataset: str, scope: str) -> pd.DataFrame:
    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = float(values.sum())
    rows = []
    for frac in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        n = max(1, int(round(len(values) * frac)))
        rows.append(
            {
                "dataset": dataset,
                "scope": scope,
                "top_row_frac": frac,
                "rows": n,
                "unit_sum": float(values[:n].sum()),
                "unit_share": float(values[:n].sum() / total) if total > 0 else 0.0,
                "min_units_in_top_bucket": float(values[n - 1]) if len(values) else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def summary(values: np.ndarray, dataset: str, scope: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "dataset": dataset,
        "scope": scope,
        "rows": int(len(values)),
        "zero_rows": int((values <= 0).sum()),
        "nonzero_rows": int((values > 0).sum()),
        "zero_rate": float((values <= 0).mean()),
        "nonzero_rate": float((values > 0).mean()),
        "unit_sum": float(values.sum()),
        "mean_units_all_rows": float(values.mean()) if len(values) else 0.0,
        "mean_units_nonzero_rows": float(values[values > 0].mean()) if (values > 0).any() else 0.0,
        "p50_all_rows": float(np.quantile(values, 0.50)) if len(values) else 0.0,
        "p90_all_rows": float(np.quantile(values, 0.90)) if len(values) else 0.0,
        "p95_all_rows": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "p99_all_rows": float(np.quantile(values, 0.99)) if len(values) else 0.0,
        "p50_nonzero": float(np.quantile(values[values > 0], 0.50)) if (values > 0).any() else 0.0,
        "p90_nonzero": float(np.quantile(values[values > 0], 0.90)) if (values > 0).any() else 0.0,
        "p95_nonzero": float(np.quantile(values[values > 0], 0.95)) if (values > 0).any() else 0.0,
        "p99_nonzero": float(np.quantile(values[values > 0], 0.99)) if (values > 0).any() else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare July holdout predicted unit deciles against labeled Weeklies sold units.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--scoring-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=NEW_DIR / "output" / "modeling_datasets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scoring_dir = args.scoring_dir or latest_scoring_dir(args.output_root)
    output_dir = scoring_dir / "decile_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading July holdout predictions")
    holdout_path = scoring_dir / "july_holdout_weeklies_predictions.parquet"
    holdout = pd.read_parquet(holdout_path, columns=["july_holdout_prediction", "positive_regressor_prediction"])

    log("Loading current labeled Weeklies sold units from database")
    labeled = run_psql_csv(args.db_uri, labeled_query())

    train_values = None
    train_path = args.dataset_dir / "weeklies" / "classifier_all.parquet"
    if train_path.exists():
        log("Loading model train split sold units from classifier_all parquet")
        train_frame = pd.read_parquet(train_path, columns=["split", "sales_target"])
        train_values = train_frame.loc[train_frame["split"] == "train", "sales_target"].to_numpy(dtype=np.float64)

    datasets: list[tuple[str, str, np.ndarray]] = [
        ("full_labeled_actual", "all_rows", labeled["sold_units"].to_numpy(dtype=np.float64)),
        ("july_holdout_final_prediction", "all_rows", holdout["july_holdout_prediction"].to_numpy(dtype=np.float64)),
        ("july_holdout_positive_regressor_pre_gate", "all_rows", holdout["positive_regressor_prediction"].to_numpy(dtype=np.float64)),
    ]
    if train_values is not None:
        datasets.insert(1, ("model_train_split_actual", "all_rows", train_values))

    positive_datasets = []
    for dataset, _, values in datasets:
        positive_datasets.append((dataset, "positive_or_nonzero_rows", values[values > 0]))

    deciles = []
    concentrations = []
    summaries = []
    for dataset, scope, values in [*datasets, *positive_datasets]:
        summaries.append(summary(values, dataset, scope))
        deciles.append(decile_table(values, dataset, scope, ascending=True))
        concentrations.append(top_concentration(values, dataset, scope))

    decile_frame = pd.concat(deciles, ignore_index=True)
    concentration_frame = pd.concat(concentrations, ignore_index=True)
    summary_frame = pd.DataFrame.from_records(summaries)
    decile_frame.to_csv(output_dir / "unit_decile_comparison.csv", index=False)
    concentration_frame.to_csv(output_dir / "top_unit_concentration.csv", index=False)
    summary_frame.to_csv(output_dir / "unit_distribution_summary.csv", index=False)

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scoring_dir": str(scoring_dir),
        "holdout_predictions_path": str(holdout_path),
        "labeled_source": "core.fact_sale joined core.dim_product where type='Weeklies' and drawqty<>0",
        "model_train_source": str(train_path) if train_values is not None else None,
        "outputs": {
            "unit_decile_comparison": str(output_dir / "unit_decile_comparison.csv"),
            "top_unit_concentration": str(output_dir / "top_unit_concentration.csv"),
            "unit_distribution_summary": str(output_dir / "unit_distribution_summary.csv"),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    log(f"Wrote decile comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
