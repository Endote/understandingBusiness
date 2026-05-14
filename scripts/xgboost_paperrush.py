#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_ROOT_DIR = ROOT_DIR / "output" / "modeling_xgboost"
CACHE_DIR = MODEL_ROOT_DIR / "cache"
DEFAULT_CORE_FEATURES_CACHE = CACHE_DIR / "core_features.csv"
OUTPUT_DIR = MODEL_ROOT_DIR / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_URI = os.getenv("PAPERRUSH_DB_URI", "postgresql://norbert.jaworski@/paperrush?host=/tmp")
TRAIN_SAMPLE_PCT = int(os.getenv("PR_XGB_TRAIN_SAMPLE_PCT", "100"))
TRAIN_SPLIT_RATIO = float(os.getenv("PR_XGB_TRAIN_RATIO", "0.70"))
VALID_SPLIT_RATIO = float(os.getenv("PR_XGB_VALID_RATIO", "0.15"))
TEST_SPLIT_RATIO = float(os.getenv("PR_XGB_TEST_RATIO", "0.15"))
CORE_FEATURES_CSV = os.getenv("PR_XGB_CORE_FEATURES_CSV", "").strip()
SEED = int(os.getenv("PR_XGB_SEED", "42"))
N_ROUNDS = int(os.getenv("PR_XGB_N_ROUNDS", "500"))
EARLY_STOPPING = int(os.getenv("PR_XGB_EARLY_STOPPING", "30"))


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_psql_copy(query: str, destination: Path) -> None:
    copy_cmd = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
    cmd = ["psql", DB_URI, "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-c", copy_cmd]
    log(f"Exporting query result to {destination.name}")
    with destination.open("wb") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.PIPE)


def psql_scalar_lines(query: str) -> list[str]:
    cmd = ["psql", DB_URI, "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-Atqc", query]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def quote_sql_date(date_str: str) -> str:
    return f"DATE '{date_str}'"


def category_sql(column: str) -> str:
    return f"coalesce(nullif(trim({column}), ''), 'UNKNOWN')"


BASE_DEMOGRAPHIC_COLUMNS = [
    "age_19_and_under",
    "age_20_to_29",
    "age_30_to_44",
    "age_45_to_59",
    "age_60_and_over",
    "male",
    "female",
    "two_or_more_races",
    "less_than_10k",
    "between_10k_and_14k",
    "between_15k_and_24k",
    "between_25k_and_34k",
    "between_35k_and_49k",
    "between_50k_and_74k",
    "between_75k_and_99k",
    "between_100k_and_149k",
    "between_150k_and_199k",
    "income_200k_or_more",
    "less_than_9th_grade",
    "between_9th_and_12th_grade_no_diploma",
    "high_school_graduate_includes_equivalency",
    "some_college_no_degree",
    "associates_degree",
    "bachelors_degree",
    "graduate_or_professional_degree",
    "population",
    "household_type_married_couple_household",
    "household_type_cohabiting_couple_household",
    "household_type_male_householder_no_spouse_partner_present",
    "household_type_female_householder_no_spouse_partner_present",
    "family_household",
    "non_family_household",
]


def build_core_feature_query(validation_start: str, test_start: str, train_sample_pct: int) -> str:
    demo_cols = ",\n        ".join(f"dg.{col}" for col in BASE_DEMOGRAPHIC_COLUMNS)
    return f"""
with base as (
    select
        fs.store_id,
        fs.product_id,
        greatest(fs.soldqty, 0)::double precision as sales_target,
        fs.soldqty::double precision as soldqty_raw,
        fs.drawqty::double precision as drawqty,
        case when fs.drawqty > 0 then greatest(fs.soldqty, 0)::double precision / fs.drawqty else null end as sellthrough,
        case when fs.drawqty > 0 and fs.soldqty = fs.drawqty then 1 else 0 end as stockout_proxy_flag,
        case when fs.soldqty < 0 then 1 else 0 end as negative_sales_flag,
        case when fs.soldqty = 0 then 1 else 0 end as zero_sales_flag,
        greatest(fs.drawqty - greatest(fs.soldqty, 0), 0)::double precision as oversupply_units,
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
),
global_date as (
    select
        onsaledate,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate
),
global_hist as (
    select
        onsaledate,
        sum(sales_sum) over (order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from global_date
),
store_date as (
    select
        onsaledate,
        store_id,
        sum(sales_target) as sales_sum,
        count(*) as row_count,
        sum(coalesce(sellthrough, 0)) as sellthrough_sum,
        count(sellthrough) as sellthrough_obs,
        sum(zero_sales_flag) as zero_sales_sum,
        sum(stockout_proxy_flag) as stockout_sum
    from base
    group by onsaledate, store_id
),
store_hist as (
    select
        onsaledate,
        store_id,
        sum(sales_sum) over (partition by store_id order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by store_id order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count,
        sum(sellthrough_sum) over (partition by store_id order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sellthrough_sum,
        sum(sellthrough_obs) over (partition by store_id order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sellthrough_obs,
        sum(zero_sales_sum) over (partition by store_id order by onsaledate rows between unbounded preceding and 1 preceding) as prior_zero_sales_sum,
        sum(stockout_sum) over (partition by store_id order by onsaledate rows between unbounded preceding and 1 preceding) as prior_stockout_sum
    from store_date
),
store_title_date as (
    select
        onsaledate,
        store_id,
        title,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate, store_id, title
),
store_title_hist as (
    select
        onsaledate,
        store_id,
        title,
        sum(sales_sum) over (partition by store_id, title order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by store_id, title order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from store_title_date
),
store_segment_date as (
    select
        onsaledate,
        store_id,
        segment,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate, store_id, segment
),
store_segment_hist as (
    select
        onsaledate,
        store_id,
        segment,
        sum(sales_sum) over (partition by store_id, segment order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by store_id, segment order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from store_segment_date
),
store_subsegment_date as (
    select
        onsaledate,
        store_id,
        subsegment,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate, store_id, subsegment
),
store_subsegment_hist as (
    select
        onsaledate,
        store_id,
        subsegment,
        sum(sales_sum) over (partition by store_id, subsegment order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by store_id, subsegment order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from store_subsegment_date
),
store_type_date as (
    select
        onsaledate,
        store_id,
        type,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate, store_id, type
),
store_type_hist as (
    select
        onsaledate,
        store_id,
        type,
        sum(sales_sum) over (partition by store_id, type order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by store_id, type order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from store_type_date
),
chain_segment_date as (
    select
        onsaledate,
        store_chain,
        segment,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate, store_chain, segment
),
chain_segment_hist as (
    select
        onsaledate,
        store_chain,
        segment,
        sum(sales_sum) over (partition by store_chain, segment order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by store_chain, segment order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from chain_segment_date
),
title_date as (
    select
        onsaledate,
        title,
        sum(sales_target) as sales_sum,
        count(*) as row_count
    from base
    group by onsaledate, title
),
title_hist as (
    select
        onsaledate,
        title,
        sum(sales_sum) over (partition by title order by onsaledate rows between unbounded preceding and 1 preceding) as prior_sales_sum,
        sum(row_count) over (partition by title order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from title_date
)
select
    case
        when b.onsaledate < {quote_sql_date(validation_start)} then 'train'
        when b.onsaledate < {quote_sql_date(test_start)} then 'valid'
        else 'test'
    end as split,
    b.store_id,
    b.product_id,
    b.onsaledate,
    b.title,
    b.type,
    b.segment,
    b.subsegment,
    b.frequency,
    b.store_chain,
    b.region,
    b.classoftrade,
    b.sales_target,
    b.soldqty_raw,
    b.drawqty,
    b.stockout_proxy_flag,
    b.negative_sales_flag,
    b.oversupply_units,
    b.sellthrough,
    b.merchandised,
    b.facings,
    b.pockets,
    b.price,
    b.issue_length_days,
    b.onsale_month,
    b.onsale_week,
    b.onsale_dow,
    coalesce(gh.prior_sales_sum / nullif(gh.prior_row_count, 0), 0) as global_prior_avg_sales,
    coalesce(sh.prior_sales_sum / nullif(sh.prior_row_count, 0), 0) as store_prior_avg_sales,
    coalesce(sh.prior_row_count, 0) as store_prior_obs,
    coalesce(sh.prior_sellthrough_sum / nullif(sh.prior_sellthrough_obs, 0), 0) as store_prior_avg_sellthrough,
    coalesce(sh.prior_zero_sales_sum / nullif(sh.prior_row_count, 0), 0) as store_prior_zero_rate,
    coalesce(sh.prior_stockout_sum / nullif(sh.prior_row_count, 0), 0) as store_prior_stockout_rate,
    coalesce(sth.prior_sales_sum / nullif(sth.prior_row_count, 0), 0) as store_title_prior_avg_sales,
    coalesce(sth.prior_row_count, 0) as store_title_prior_obs,
    coalesce(sseh.prior_sales_sum / nullif(sseh.prior_row_count, 0), 0) as store_segment_prior_avg_sales,
    coalesce(sseh.prior_row_count, 0) as store_segment_prior_obs,
    coalesce(ssubh.prior_sales_sum / nullif(ssubh.prior_row_count, 0), 0) as store_subsegment_prior_avg_sales,
    coalesce(ssubh.prior_row_count, 0) as store_subsegment_prior_obs,
    coalesce(styh.prior_sales_sum / nullif(styh.prior_row_count, 0), 0) as store_type_prior_avg_sales,
    coalesce(styh.prior_row_count, 0) as store_type_prior_obs,
    coalesce(csh.prior_sales_sum / nullif(csh.prior_row_count, 0), 0) as chain_segment_prior_avg_sales,
    coalesce(csh.prior_row_count, 0) as chain_segment_prior_obs,
    coalesce(th.prior_sales_sum / nullif(th.prior_row_count, 0), 0) as title_prior_avg_sales,
    coalesce(th.prior_row_count, 0) as title_prior_obs,
    {", ".join(f"coalesce(b.{col}, 0) as {col}" for col in BASE_DEMOGRAPHIC_COLUMNS)}
from base b
left join global_hist gh on gh.onsaledate = b.onsaledate
left join store_hist sh on sh.onsaledate = b.onsaledate and sh.store_id = b.store_id
left join store_title_hist sth on sth.onsaledate = b.onsaledate and sth.store_id = b.store_id and sth.title = b.title
left join store_segment_hist sseh on sseh.onsaledate = b.onsaledate and sseh.store_id = b.store_id and sseh.segment = b.segment
left join store_subsegment_hist ssubh on ssubh.onsaledate = b.onsaledate and ssubh.store_id = b.store_id and ssubh.subsegment = b.subsegment
left join store_type_hist styh on styh.onsaledate = b.onsaledate and styh.store_id = b.store_id and styh.type = b.type
left join chain_segment_hist csh on csh.onsaledate = b.onsaledate and csh.store_chain = b.store_chain and csh.segment = b.segment
left join title_hist th on th.onsaledate = b.onsaledate and th.title = b.title
where
    (b.onsaledate >= {quote_sql_date(validation_start)})
    or (
        b.onsaledate < {quote_sql_date(validation_start)}
        and mod(abs(hashtextextended(concat(b.store_id::text, '|', b.product_id::text), 0)), 100) < {train_sample_pct}
    )
order by b.onsaledate, b.product_id, b.store_id
"""


@dataclass
class EncodedMatrices:
    train_matrix: sparse.csr_matrix
    valid_matrix: sparse.csr_matrix
    test_matrix: sparse.csr_matrix
    feature_names: list[str]


@dataclass
class FamilyRunResult:
    family: str
    best_iteration: int
    best_score: float
    test_metrics: dict[str, object]
    training_diagnostics: dict[str, object]
    metrics_df: pd.DataFrame
    test_predictions: pd.DataFrame
    importance_df: pd.DataFrame
    grouped_importance_df: pd.DataFrame


def export_query_to_dataframe(query: str, file_stub: str, csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = OUTPUT_DIR / f"{file_stub}.csv"
    run_psql_copy(query, csv_path)
    log(f"Loading {csv_path.name} into pandas")
    return pd.read_csv(csv_path, parse_dates=["onsaledate"])


def load_existing_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    log(f"Reusing precomputed features from {path}")
    return pd.read_csv(path, parse_dates=["onsaledate"])


def prepare_frames(core_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_df = core_df.loc[(core_df["sales_target"] > 0) & (core_df["drawqty"] > 0)].copy()

    train_df = core_df.loc[core_df["split"] == "train"].copy()
    valid_df = core_df.loc[core_df["split"] == "valid"].copy()
    test_df = core_df.loc[core_df["split"] == "test"].copy()

    categorical_cols = [
        "title",
        "type",
        "segment",
        "subsegment",
        "frequency",
        "store_chain",
        "region",
        "classoftrade",
    ]
    passthrough_cols = {"split", "store_id", "product_id", "onsaledate", "sales_target", "soldqty_raw", "drawqty", "stockout_proxy_flag", "negative_sales_flag", "oversupply_units", "sellthrough"}

    for frame in (train_df, valid_df, test_df):
        for col in categorical_cols:
            frame[col] = frame[col].astype("string").fillna("UNKNOWN").astype(str)

    for col in categorical_cols:
        levels = sorted(set(train_df[col]).union(valid_df[col]).union(test_df[col]))
        train_df[col] = pd.Categorical(train_df[col], categories=levels)
        valid_df[col] = pd.Categorical(valid_df[col], categories=levels)
        test_df[col] = pd.Categorical(test_df[col], categories=levels)

    numeric_cols = [
        col for col in train_df.columns
        if col not in passthrough_cols and col not in categorical_cols
    ]

    for col in numeric_cols:
        median_value = float(train_df[col].median()) if not pd.isna(train_df[col].median()) else 0.0
        train_df[col] = train_df[col].fillna(median_value)
        valid_df[col] = valid_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    return train_df, valid_df, test_df


def encode_features(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> EncodedMatrices:
    categorical_cols = [
        "title",
        "type",
        "segment",
        "subsegment",
        "frequency",
        "store_chain",
        "region",
        "classoftrade",
    ]
    exclude_cols = {
        "split",
        "store_id",
        "product_id",
        "onsaledate",
        "sales_target",
        "soldqty_raw",
        "drawqty",
        "stockout_proxy_flag",
        "negative_sales_flag",
        "oversupply_units",
        "sellthrough",
    }
    numeric_cols = [col for col in train_df.columns if col not in exclude_cols and col not in categorical_cols]

    train_num = sparse.csr_matrix(train_df[numeric_cols].to_numpy(dtype=np.float32))
    valid_num = sparse.csr_matrix(valid_df[numeric_cols].to_numpy(dtype=np.float32))
    test_num = sparse.csr_matrix(test_df[numeric_cols].to_numpy(dtype=np.float32))

    combined = pd.concat(
        [
            train_df[categorical_cols],
            valid_df[categorical_cols],
            test_df[categorical_cols],
        ],
        axis=0,
    )
    combined_dummies = pd.get_dummies(combined, columns=categorical_cols, sparse=True, dtype=np.float32)
    train_cat = sparse.csr_matrix(combined_dummies.iloc[: len(train_df)].sparse.to_coo())
    valid_cat = sparse.csr_matrix(combined_dummies.iloc[len(train_df): len(train_df) + len(valid_df)].sparse.to_coo())
    test_cat = sparse.csr_matrix(combined_dummies.iloc[len(train_df) + len(valid_df):].sparse.to_coo())

    feature_names = numeric_cols + list(combined_dummies.columns)
    return EncodedMatrices(
        train_matrix=sparse.hstack([train_num, train_cat], format="csr"),
        valid_matrix=sparse.hstack([valid_num, valid_cat], format="csr"),
        test_matrix=sparse.hstack([test_num, test_cat], format="csr"),
        feature_names=feature_names,
    )


def build_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    # DrawQty affects observed sales through stockouts and over-allocation.
    # We keep SoldQty-based target, but reduce weight on likely censored stockout rows
    # and slightly upweight high-draw observations because they carry more profit exposure.
    draw_component = np.clip(np.log1p(frame["drawqty"].fillna(0).to_numpy(dtype=np.float32)), 0.0, 3.5)
    stockout_penalty = np.where(frame["stockout_proxy_flag"].to_numpy(dtype=np.int8) == 1, 0.75, 1.0)
    negative_penalty = np.where(frame["negative_sales_flag"].to_numpy(dtype=np.int8) == 1, 0.60, 1.0)
    return (1.0 + 0.20 * draw_component) * stockout_penalty * negative_penalty


def safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom == 0:
        return math.nan
    return float(np.abs(y_true - y_pred).sum() / denom)


def build_metrics(frame: pd.DataFrame, prediction_col: str, dataset_name: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    slices = {
        "all": np.ones(len(frame), dtype=bool),
        "stockout_proxy": frame["stockout_proxy_flag"].eq(1).to_numpy(),
        "non_stockout": frame["stockout_proxy_flag"].eq(0).to_numpy(),
    }
    for slice_name, mask in slices.items():
        if mask.sum() == 0:
            continue
        y_true = frame.loc[mask, "sales_target"].to_numpy(dtype=np.float32)
        y_pred = frame.loc[mask, prediction_col].to_numpy(dtype=np.float32)
        records.append(
            {
                "slice": slice_name,
                "dataset": dataset_name,
                "rows": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "wape": safe_wape(y_true, y_pred),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_baseline(frame: pd.DataFrame) -> np.ndarray:
    baseline = frame["store_title_prior_avg_sales"].to_numpy(dtype=np.float32)
    fallback_order = [
        "store_segment_prior_avg_sales",
        "store_prior_avg_sales",
        "chain_segment_prior_avg_sales",
        "title_prior_avg_sales",
        "global_prior_avg_sales",
    ]
    needs_fill = baseline <= 0
    for col in fallback_order:
        replacement = frame[col].to_numpy(dtype=np.float32)
        baseline = np.where(needs_fill, replacement, baseline)
        needs_fill = baseline <= 0
    return np.clip(baseline, 0.0, None)


def compute_split_dates(train_ratio: float, valid_ratio: float, test_ratio: float) -> dict[str, object]:
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    dates = psql_scalar_lines("select distinct onsaledate::text from core.dim_product order by 1")
    if len(dates) < 3:
        raise ValueError("Need at least 3 distinct issue dates for a train/validation/test split")

    n_dates = len(dates)
    train_dates = max(1, round(n_dates * train_ratio))
    valid_dates = max(1, round(n_dates * valid_ratio))
    if train_dates + valid_dates >= n_dates:
        valid_dates = max(1, n_dates - train_dates - 1)
    test_dates = n_dates - train_dates - valid_dates
    if test_dates < 1:
        test_dates = 1
        if valid_dates > 1:
            valid_dates -= 1
        else:
            train_dates -= 1

    validation_start = dates[train_dates]
    test_start = dates[train_dates + valid_dates]
    return {
        "all_dates": dates,
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "train_start": dates[0],
        "train_end": dates[train_dates - 1],
        "validation_start": validation_start,
        "validation_end": dates[train_dates + valid_dates - 1],
        "test_start": test_start,
        "test_end": dates[-1],
    }


def summarize_existing_split(core_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split_name in ["train", "valid", "test"]:
        split_dates = sorted(core_df.loc[core_df["split"] == split_name, "onsaledate"].dt.strftime("%Y-%m-%d").unique())
        if not split_dates:
            raise ValueError(f"Cached feature file is missing split {split_name}")
        summary[f"{split_name}_dates"] = len(split_dates)
        summary[f"{split_name}_start"] = split_dates[0]
        summary[f"{split_name}_end"] = split_dates[-1]
    summary["validation_start"] = summary["valid_start"]
    summary["validation_end"] = summary["valid_end"]
    return summary


def feature_group(feature_name: str) -> str:
    if feature_name.startswith(("store_title_", "title_prior_")):
        return "title_history"
    if feature_name.startswith(("store_segment_", "store_subsegment_", "chain_segment_")):
        return "category_history"
    if feature_name.startswith(("store_prior_", "store_type_")):
        return "store_history"
    if feature_name in {"price", "issue_length_days", "onsale_month", "onsale_week", "onsale_dow"}:
        return "product_calendar"
    if feature_name in {"merchandised", "facings", "pockets"}:
        return "store_attributes"
    if feature_name in BASE_DEMOGRAPHIC_COLUMNS:
        return "demographics"
    return "categoricals"


def train_family_model(
    family: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> FamilyRunResult:
    family_train_df = train_df.loc[train_df["type"] == family].copy()
    family_valid_df = valid_df.loc[valid_df["type"] == family].copy()
    family_test_df = test_df.loc[test_df["type"] == family].copy()

    if family_train_df.empty or family_valid_df.empty or family_test_df.empty:
        raise ValueError(f"Family {family} does not have non-empty train/validation/test splits")

    family_train_df, family_valid_df, family_test_df = prepare_frames(
        pd.concat([family_train_df, family_valid_df, family_test_df], ignore_index=True)
    )
    matrices = encode_features(family_train_df, family_valid_df, family_test_df)

    train_labels = np.log1p(family_train_df["sales_target"].to_numpy(dtype=np.float32))
    valid_labels = np.log1p(family_valid_df["sales_target"].to_numpy(dtype=np.float32))
    test_labels = np.log1p(family_test_df["sales_target"].to_numpy(dtype=np.float32))

    train_weights = build_sample_weights(family_train_df)
    valid_weights = build_sample_weights(family_valid_df)
    test_weights = build_sample_weights(family_test_df)

    dtrain = xgb.DMatrix(matrices.train_matrix, label=train_labels, weight=train_weights, feature_names=matrices.feature_names)
    dvalid = xgb.DMatrix(matrices.valid_matrix, label=valid_labels, weight=valid_weights, feature_names=matrices.feature_names)
    dtest = xgb.DMatrix(matrices.test_matrix, label=test_labels, weight=test_weights, feature_names=matrices.feature_names)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": SEED,
        "nthread": max(os.cpu_count() - 1, 1),
    }

    log(f"Training XGBoost model for {family}")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=N_ROUNDS,
        evals=[(dtrain, f"{family.lower()}_train"), (dvalid, f"{family.lower()}_valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=25,
    )

    family_valid_df["baseline_pred"] = compute_baseline(family_valid_df)
    family_test_df["baseline_pred"] = compute_baseline(family_test_df)
    family_valid_df["xgb_pred"] = np.clip(
        np.expm1(booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))),
        0.0,
        None,
    )
    family_test_df["xgb_pred"] = np.clip(
        np.expm1(booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))),
        0.0,
        None,
    )

    xgb_metrics = pd.concat(
        [
            build_metrics(family_valid_df, "xgb_pred", "validation"),
            build_metrics(family_test_df, "xgb_pred", "test"),
        ],
        ignore_index=True,
    )
    xgb_metrics.insert(0, "model", "xgboost")
    xgb_metrics.insert(0, "family", family)

    baseline_metrics = pd.concat(
        [
            build_metrics(family_valid_df, "baseline_pred", "validation"),
            build_metrics(family_test_df, "baseline_pred", "test"),
        ],
        ignore_index=True,
    )
    baseline_metrics.insert(0, "model", "historical_fallback")
    baseline_metrics.insert(0, "family", family)

    metrics_df = pd.concat([baseline_metrics, xgb_metrics], ignore_index=True)

    importance_map = booster.get_score(importance_type="gain")
    importance_df = (
        pd.DataFrame(
            [
                {"family": family, "feature": feature, "gain": gain, "feature_group": feature_group(feature)}
                for feature, gain in importance_map.items()
            ]
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    grouped_importance_df = (
        importance_df.groupby(["family", "feature_group"], as_index=False)["gain"]
        .sum()
        .sort_values(["family", "gain"], ascending=[True, False])
    )

    raw_test_row = xgb_metrics[(xgb_metrics["slice"] == "all") & (xgb_metrics["dataset"] == "test")].iloc[0]

    test_predictions = family_test_df[
        [
            "store_id",
            "product_id",
            "onsaledate",
            "type",
            "sales_target",
            "drawqty",
            "stockout_proxy_flag",
            "baseline_pred",
            "xgb_pred",
        ]
    ].copy()
    test_predictions.insert(0, "family", family)

    importance_df.to_csv(OUTPUT_DIR / f"feature_importance_gain_{family.lower()}.csv", index=False)
    grouped_importance_df.to_csv(OUTPUT_DIR / f"feature_group_importance_gain_{family.lower()}.csv", index=False)

    test_metrics = {
        "metric_scale": "raw_copy_counts",
        "source_dataset": "test",
        "mae": float(raw_test_row["mae"]),
        "rmse": float(raw_test_row["rmse"]),
        "wape": float(raw_test_row["wape"]),
    }
    training_diagnostics = {
        "best_iteration": int(booster.best_iteration),
        "early_stopping_metric": "rmse_on_log1p_target",
        "early_stopping_best_score": float(booster.best_score),
    }

    return FamilyRunResult(
        family=family,
        best_iteration=int(booster.best_iteration),
        best_score=float(booster.best_score),
        test_metrics=test_metrics,
        training_diagnostics=training_diagnostics,
        metrics_df=metrics_df.assign(
            early_stopping_metric="rmse_on_log1p_target",
            prediction_metric_scale="raw_copy_counts",
        ),
        test_predictions=test_predictions,
        importance_df=importance_df,
        grouped_importance_df=grouped_importance_df,
    )


def main() -> int:
    if CORE_FEATURES_CSV:
        core_df = load_existing_dataframe(CORE_FEATURES_CSV)
        split_info = summarize_existing_split(core_df)
    elif DEFAULT_CORE_FEATURES_CACHE.exists():
        core_df = load_existing_dataframe(str(DEFAULT_CORE_FEATURES_CACHE))
        split_info = summarize_existing_split(core_df)
    else:
        split_info = compute_split_dates(TRAIN_SPLIT_RATIO, VALID_SPLIT_RATIO, TEST_SPLIT_RATIO)
        core_query = build_core_feature_query(
            validation_start=split_info["validation_start"],
            test_start=split_info["test_start"],
            train_sample_pct=TRAIN_SAMPLE_PCT,
        )
        core_df = export_query_to_dataframe(core_query, "core_features", DEFAULT_CORE_FEATURES_CACHE)

    log(
        "Using chronological split: "
        f"train {split_info['train_start']} to {split_info['train_end']} "
        f"({split_info['train_dates']} dates), "
        f"validation {split_info['validation_start']} to {split_info['validation_end']} "
        f"({split_info['valid_dates']} dates), "
        f"test {split_info['test_start']} to {split_info['test_end']} "
        f"({split_info['test_dates']} dates)"
    )

    train_df, valid_df, test_df = prepare_frames(core_df)
    family_results = []
    for family in ["Weeklies", "SIP"]:
        family_results.append(train_family_model(family, train_df, valid_df, test_df))

    metrics_df = pd.concat([result.metrics_df for result in family_results], ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    test_output = pd.concat([result.test_predictions for result in family_results], ignore_index=True)
    test_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    summary = {
        "db_uri": DB_URI,
        "train_sample_pct": TRAIN_SAMPLE_PCT,
        "train_ratio": TRAIN_SPLIT_RATIO,
        "validation_ratio": VALID_SPLIT_RATIO,
        "test_ratio": TEST_SPLIT_RATIO,
        "train_start": split_info["train_start"],
        "train_end": split_info["train_end"],
        "validation_start": split_info["validation_start"],
        "validation_end": split_info["validation_end"],
        "test_start": split_info["test_start"],
        "test_end": split_info["test_end"],
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "family_runs": {
            result.family: {
                "train_rows": int((train_df["type"] == result.family).sum()),
                "valid_rows": int((valid_df["type"] == result.family).sum()),
                "test_rows": int((test_df["type"] == result.family).sum()),
                "test_metrics": result.test_metrics,
            }
            for result in family_results
        },
        "training_diagnostics": {
            result.family: result.training_diagnostics
            for result in family_results
        },
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"Finished. Outputs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
