#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_modeling_datasets import (
    BASE_DEMOGRAPHIC_COLUMNS,
    COMMON_NUMERIC_FEATURES,
    DEFAULT_DB_URI,
    DEFAULT_OUTPUT_DIR,
    FAMILIES,
    FAMILY_CATEGORICAL_FEATURES,
    add_completed_prior_features,
    add_derived_numeric_features,
    add_embedding_features,
    assign_split,
    build_expanding_folds,
    category_sql,
    log,
    require_parquet_engine,
    run_psql_copy,
    split_dates,
)


STAGE = "classifier_all"


def build_classifier_all_query(family: str) -> str:
    demo_cols = ",\n        ".join(f"dg.{col}" for col in BASE_DEMOGRAPHIC_COLUMNS)
    demo_select = ",\n    ".join(f"coalesce(t.{col}, 0) as {col}" for col in BASE_DEMOGRAPHIC_COLUMNS)
    return f"""
with base_all as (
    select
        fs.store_id,
        fs.product_id,
        greatest(fs.soldqty, 0)::double precision as sales_target,
        fs.soldqty::double precision as soldqty_raw,
        fs.drawqty::double precision as drawqty,
        case when fs.soldqty > 0 then 1 else 0 end as positive_sale_flag,
        case when fs.soldqty < 0 then 1 else 0 end as negative_sales_flag,
        case when fs.drawqty > 0 and fs.soldqty = fs.drawqty then 1 else 0 end as stockout_proxy_flag,
        {category_sql("dp.title")} as title,
        {category_sql("dp.type")} as type,
        {category_sql("dp.segment")} as segment,
        {category_sql("dp.subsegment")} as subsegment,
        {category_sql("dp.frequency")} as frequency,
        dp.onsaledate,
        dp.offsaledate,
        coalesce(dp.price, 0)::double precision as price,
        coalesce((dp.offsaledate - dp.onsaledate), 0)::double precision as issue_length_days,
        extract(month from dp.onsaledate)::int as onsale_month,
        'month_' || lpad(extract(month from dp.onsaledate)::int::text, 2, '0') as onsale_month_cat,
        extract(week from dp.onsaledate)::int as onsale_week,
        extract(dow from dp.onsaledate)::int as onsale_dow,
        {category_sql("ds.store_chain")} as store_chain,
        {category_sql("ds.region")} as region,
        {category_sql("ds.classoftrade")} as classoftrade,
        case when ds.merchandised then 1 else 0 end as merchandised,
        coalesce(ds.facings, 0)::double precision as facings,
        coalesce(ds.pockets, 0)::double precision as pockets,
        {demo_cols}
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    join core.dim_store ds on ds.store_id = fs.store_id
    left join core.demographic dg on dg.postal_code = ds.postal_code
    where dp.type = '{family}'
      and fs.drawqty <> 0
)
select
    t.store_id,
    t.product_id,
    t.onsaledate,
    t.offsaledate,
    t.title,
    t.type,
    t.segment,
    t.subsegment,
    t.frequency,
    t.store_chain,
    t.region,
    t.classoftrade,
    t.sales_target,
    t.soldqty_raw,
    t.drawqty,
    t.positive_sale_flag,
    t.negative_sales_flag,
    t.stockout_proxy_flag,
    t.price,
    t.issue_length_days,
    t.onsale_month,
    t.onsale_month_cat,
    t.onsale_week,
    t.onsale_dow,
    t.merchandised,
    t.facings,
    t.pockets,
    {demo_select}
from base_all t
order by t.onsaledate, t.product_id, t.store_id
"""


def classifier_feature_contract(family: str) -> dict[str, object]:
    return {
        "family": family,
        "stage": STAGE,
        "row_filter": "DrawQty != 0",
        "target": "positive_sale_flag",
        "id_columns": [
            "store_id",
            "product_id",
            "onsaledate",
            "offsaledate",
            "split",
            "sales_target",
            "soldqty_raw",
            "drawqty",
            "positive_sale_flag",
            "negative_sales_flag",
            "stockout_proxy_flag",
        ],
        "categorical_features": FAMILY_CATEGORICAL_FEATURES[family],
        "numeric_features": COMMON_NUMERIC_FEATURES,
        "excluded_scoring_unavailable_columns": ["drawqty"],
    }


def build_family_classifier_dataset(
    family: str,
    db_uri: str,
    output_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    folds: int,
    valid_dates_per_fold: int,
    min_train_dates: int,
) -> dict[str, object]:
    query = build_classifier_all_query(family)
    with tempfile.TemporaryDirectory(prefix="paperrush_incidence_") as tmp_dir:
        csv_path = Path(tmp_dir) / f"{family.lower()}_{STAGE}.csv"
        log(f"Exporting {family} all-row incidence classifier dataset from Postgres")
        run_psql_copy(db_uri, query, csv_path)
        log(f"Loading {family} all-row CSV export")
        frame = pd.read_csv(csv_path, parse_dates=["onsaledate"])

    dates = sorted(frame["onsaledate"].dt.strftime("%Y-%m-%d").unique())
    split_info = split_dates(dates, train_ratio, valid_ratio, test_ratio)
    frame = assign_split(frame, split_info)
    family_dir = output_dir / family.lower()
    family_dir.mkdir(parents=True, exist_ok=True)
    frame, prior_summary = add_completed_prior_features(frame, family, db_uri)
    frame, embedding_summary = add_embedding_features(frame, family, db_uri, family_dir, stage=STAGE)
    frame = add_derived_numeric_features(frame)

    parquet_path = family_dir / f"{STAGE}.parquet"
    frame.to_parquet(parquet_path, index=False)
    contract = classifier_feature_contract(family)
    contract_path = family_dir / f"feature_contract_{STAGE}.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    folds_payload = build_expanding_folds(
        split_info["train_dates"] + split_info["validation_dates"],
        folds=folds,
        valid_dates_per_fold=valid_dates_per_fold,
        min_train_dates=min_train_dates,
    )
    summary = {
        "family": family,
        "stage": STAGE,
        "parquet_path": str(parquet_path),
        "feature_contract_path": str(contract_path),
        "rows": int(len(frame)),
        "dates": len(dates),
        "train_rows": int((frame["split"] == "train").sum()),
        "valid_rows": int((frame["split"] == "valid").sum()),
        "test_rows": int((frame["split"] == "test").sum()),
        "positive_rate": float(frame["positive_sale_flag"].mean()),
        "split_positive_rates": {
            split: float(frame.loc[frame["split"] == split, "positive_sale_flag"].mean())
            for split in ("train", "valid", "test")
        },
        "sales_target_summary": {
            "mean": float(frame["sales_target"].mean()),
            "median": float(frame["sales_target"].median()),
            "p90": float(frame["sales_target"].quantile(0.90)),
            "p99": float(frame["sales_target"].quantile(0.99)),
        },
        "completed_prior_features": prior_summary,
        "embedding_features": embedding_summary,
        "split_dates": split_info,
        "folds": folds_payload,
    }
    (family_dir / f"manifest_{STAGE}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-row positive-sale incidence classifier datasets.")
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="Weeklies")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--valid-dates-per-fold", type=int, default=4)
    parser.add_argument("--min-train-dates", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    require_parquet_engine()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = FAMILIES if args.family == "all" else (args.family,)
    summaries = [
        build_family_classifier_dataset(
            family=family,
            db_uri=args.db_uri,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            folds=args.folds,
            valid_dates_per_fold=args.valid_dates_per_fold,
            min_train_dates=args.min_train_dates,
        )
        for family in families
    ]
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "db_uri": args.db_uri,
        "stage": STAGE,
        "families": summaries,
    }
    (args.output_dir / f"split_manifest_{STAGE}.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    log(f"Wrote incidence classifier datasets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
