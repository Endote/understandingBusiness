#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import defaultdict
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
CV_FOLDS = int(os.getenv("PR_XGB_CV_FOLDS", "3"))
CV_VALID_DATES = int(os.getenv("PR_XGB_CV_VALID_DATES", "4"))
CV_MIN_TRAIN_DATES = int(os.getenv("PR_XGB_CV_MIN_TRAIN_DATES", "12"))
CV_N_ROUNDS = int(os.getenv("PR_XGB_CV_N_ROUNDS", str(min(N_ROUNDS, 200))))
SHAP_MAX_ROWS = int(os.getenv("PR_XGB_SHAP_MAX_ROWS", "5000"))
CALIBRATION_BINS = int(os.getenv("PR_XGB_CALIBRATION_BINS", "10"))
UNKNOWN_CATEGORY_TOKEN = "__UNKNOWN_CATEGORY__"
MISSING_CATEGORY_TOKEN = "UNKNOWN"
FAMILIES = ["Weeklies", "SIP"]

CATEGORICAL_COLS = [
    "title",
    "type",
    "segment",
    "subsegment",
    "frequency",
    "store_chain",
    "region",
    "classoftrade",
]

PASSTHROUGH_COLS = {
    "split",
    "store_id",
    "product_id",
    "onsaledate",
    "sales_target",
    "soldqty_raw",
    "drawqty",
    "stockout_proxy_flag",
    "negative_sales_flag",
    "zero_sales_flag",
    "oversupply_units",
    "sellthrough",
}


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
        b.zero_sales_flag,
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
class PreprocessingArtifact:
    categorical_cols: list[str]
    numeric_cols: list[str]
    category_levels: dict[str, list[str]]
    numeric_medians: dict[str, float]
    unknown_category_token: str
    missing_category_token: str


@dataclass
class FamilyRunResult:
    family: str
    best_iteration: int
    best_score: float
    test_metrics: dict[str, object]
    training_diagnostics: dict[str, object]
    metrics_df: pd.DataFrame
    rolling_metrics_df: pd.DataFrame
    rolling_summary_df: pd.DataFrame
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


def normalize_core_dataframe(core_df: pd.DataFrame) -> pd.DataFrame:
    if "zero_sales_flag" not in core_df.columns:
        core_df = core_df.copy()
        core_df["zero_sales_flag"] = (core_df["sales_target"] == 0).astype(np.int8)
    return core_df


def split_modeling_frames(core_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    before_rows = int(len(core_df))
    modeling_df = core_df.loc[core_df["drawqty"] != 0].copy()
    dropped_zero_draw = before_rows - int(len(modeling_df))
    split_counts = {
        "input_rows": before_rows,
        "dropped_zero_draw_rows": dropped_zero_draw,
        "modeling_rows": int(len(modeling_df)),
        "zero_sales_rows_kept": int((modeling_df["sales_target"] == 0).sum()),
    }
    train_df = modeling_df.loc[modeling_df["split"] == "train"].copy()
    valid_df = modeling_df.loc[modeling_df["split"] == "valid"].copy()
    test_df = modeling_df.loc[modeling_df["split"] == "test"].copy()
    return train_df, valid_df, test_df, split_counts


def fit_preprocessing(train_df: pd.DataFrame) -> PreprocessingArtifact:
    category_levels: dict[str, list[str]] = {}
    prepared_train = train_df.copy()
    for col in CATEGORICAL_COLS:
        prepared_train[col] = prepared_train[col].astype("string").fillna(MISSING_CATEGORY_TOKEN).astype(str)
        levels = sorted(set(prepared_train[col]))
        if UNKNOWN_CATEGORY_TOKEN not in levels:
            levels.append(UNKNOWN_CATEGORY_TOKEN)
        category_levels[col] = levels

    numeric_cols = [
        col for col in train_df.columns
        if col not in PASSTHROUGH_COLS and col not in CATEGORICAL_COLS
    ]
    numeric_medians = {}
    for col in numeric_cols:
        median_value = train_df[col].median()
        numeric_medians[col] = float(median_value) if not pd.isna(median_value) else 0.0

    return PreprocessingArtifact(
        categorical_cols=list(CATEGORICAL_COLS),
        numeric_cols=numeric_cols,
        category_levels=category_levels,
        numeric_medians=numeric_medians,
        unknown_category_token=UNKNOWN_CATEGORY_TOKEN,
        missing_category_token=MISSING_CATEGORY_TOKEN,
    )


def apply_preprocessing(frame: pd.DataFrame, artifact: PreprocessingArtifact) -> tuple[pd.DataFrame, dict[str, object]]:
    prepared = frame.copy()
    unknown_counts: dict[str, int] = {}
    for col in artifact.categorical_cols:
        levels = artifact.category_levels[col]
        values = prepared[col].astype("string").fillna(artifact.missing_category_token).astype(str)
        known_mask = values.isin(levels)
        unknown_counts[col] = int((~known_mask).sum())
        values = values.where(known_mask, artifact.unknown_category_token)
        prepared[col] = pd.Categorical(values, categories=levels)

    for col in artifact.numeric_cols:
        prepared[col] = prepared[col].fillna(artifact.numeric_medians[col])

    diagnostics = {
        "rows": int(len(prepared)),
        "unknown_category_counts": unknown_counts,
        "unknown_category_rates": {
            col: (count / len(prepared) if len(prepared) else 0.0)
            for col, count in unknown_counts.items()
        },
    }
    return prepared, diagnostics


def write_preprocessing_artifact(
    family: str,
    artifact: PreprocessingArtifact,
    diagnostics: dict[str, object],
) -> None:
    payload = {
        "family": family,
        "fit_scope": "training_split_only",
        "unknown_category_handling": "Values absent from the training split are mapped to __UNKNOWN_CATEGORY__.",
        "categorical_cols": artifact.categorical_cols,
        "numeric_cols": artifact.numeric_cols,
        "category_levels": artifact.category_levels,
        "numeric_medians": artifact.numeric_medians,
        "diagnostics": diagnostics,
    }
    path = OUTPUT_DIR / f"preprocessing_{family.lower()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_family_frames(
    family: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocessingArtifact, dict[str, object]]:
    artifact = fit_preprocessing(train_df)
    prepared_train, train_diag = apply_preprocessing(train_df, artifact)
    prepared_valid, valid_diag = apply_preprocessing(valid_df, artifact)
    prepared_test, test_diag = apply_preprocessing(test_df, artifact)
    diagnostics = {
        "train": train_diag,
        "validation": valid_diag,
        "test": test_diag,
    }
    if persist:
        write_preprocessing_artifact(family, artifact, diagnostics)
    return prepared_train, prepared_valid, prepared_test, artifact, diagnostics


def encode_categorical_frame(frame: pd.DataFrame, artifact: PreprocessingArtifact) -> sparse.csr_matrix:
    dummies = pd.get_dummies(
        frame[artifact.categorical_cols],
        columns=artifact.categorical_cols,
        sparse=True,
        dtype=np.float32,
    )
    expected_cols = [
        f"{col}_{level}"
        for col in artifact.categorical_cols
        for level in artifact.category_levels[col]
    ]
    missing_cols = [col for col in expected_cols if col not in dummies.columns]
    for col in missing_cols:
        dummies[col] = pd.arrays.SparseArray(np.zeros(len(frame), dtype=np.float32), fill_value=0.0)
    dummies = dummies[expected_cols]
    return sparse.csr_matrix(dummies.sparse.to_coo())


def encode_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifact: PreprocessingArtifact,
) -> EncodedMatrices:
    train_num = sparse.csr_matrix(train_df[artifact.numeric_cols].to_numpy(dtype=np.float32))
    valid_num = sparse.csr_matrix(valid_df[artifact.numeric_cols].to_numpy(dtype=np.float32))
    test_num = sparse.csr_matrix(test_df[artifact.numeric_cols].to_numpy(dtype=np.float32))

    train_cat = encode_categorical_frame(train_df, artifact)
    valid_cat = encode_categorical_frame(valid_df, artifact)
    test_cat = encode_categorical_frame(test_df, artifact)

    categorical_feature_names = [
        f"{col}_{level}"
        for col in artifact.categorical_cols
        for level in artifact.category_levels[col]
    ]
    feature_names = artifact.numeric_cols + categorical_feature_names
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
    draw_component = np.clip(np.log1p(np.clip(frame["drawqty"].fillna(0).to_numpy(dtype=np.float32), 0.0, None)), 0.0, 3.5)
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
        "zero_sales": frame["sales_target"].eq(0).to_numpy(),
        "positive_sales": frame["sales_target"].gt(0).to_numpy(),
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


def build_xgb_params() -> dict[str, object]:
    return {
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


def train_booster(
    family: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    matrices: EncodedMatrices,
    num_boost_round: int,
    early_stopping_rounds: int,
    log_label: str,
) -> tuple[xgb.Booster, xgb.DMatrix, xgb.DMatrix]:
    train_labels = np.log1p(train_df["sales_target"].to_numpy(dtype=np.float32))
    valid_labels = np.log1p(valid_df["sales_target"].to_numpy(dtype=np.float32))

    dtrain = xgb.DMatrix(
        matrices.train_matrix,
        label=train_labels,
        weight=build_sample_weights(train_df),
        feature_names=matrices.feature_names,
    )
    dvalid = xgb.DMatrix(
        matrices.valid_matrix,
        label=valid_labels,
        weight=build_sample_weights(valid_df),
        feature_names=matrices.feature_names,
    )

    log(f"Training XGBoost model for {family} ({log_label})")
    booster = xgb.train(
        params=build_xgb_params(),
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, f"{family.lower()}_{log_label}_train"), (dvalid, f"{family.lower()}_{log_label}_valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=25,
    )
    return booster, dtrain, dvalid


def predict_raw_units(booster: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    return np.clip(
        np.expm1(booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))),
        0.0,
        None,
    )


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


def metric_record(frame: pd.DataFrame, prediction_col: str, dataset_name: str, slice_name: str) -> dict[str, object]:
    y_true = frame["sales_target"].to_numpy(dtype=np.float32)
    y_pred = frame[prediction_col].to_numpy(dtype=np.float32)
    residual = y_pred - y_true
    return {
        "dataset": dataset_name,
        "slice": slice_name,
        "rows": int(len(frame)),
        "actual_mean": float(np.mean(y_true)) if len(frame) else math.nan,
        "prediction_mean": float(np.mean(y_pred)) if len(frame) else math.nan,
        "bias_mean_pred_minus_actual": float(np.mean(residual)) if len(frame) else math.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(frame) else math.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(frame) else math.nan,
        "wape": safe_wape(y_true, y_pred) if len(frame) else math.nan,
        "zero_sales_rate": float((frame["sales_target"] == 0).mean()) if len(frame) else math.nan,
        "stockout_proxy_rate": float(frame["stockout_proxy_flag"].mean()) if len(frame) else math.nan,
    }


def build_named_slice_diagnostics(frame: pd.DataFrame, prediction_col: str, dataset_name: str) -> pd.DataFrame:
    slice_masks = {
        "all": np.ones(len(frame), dtype=bool),
        "zero_sales": frame["sales_target"].eq(0).to_numpy(),
        "positive_sales": frame["sales_target"].gt(0).to_numpy(),
        "stockout_proxy": frame["stockout_proxy_flag"].eq(1).to_numpy(),
        "non_stockout": frame["stockout_proxy_flag"].eq(0).to_numpy(),
        "cold_store": frame["store_prior_obs"].eq(0).to_numpy(),
        "sparse_store_lt_5": frame["store_prior_obs"].lt(5).to_numpy(),
        "cold_store_title": frame["store_title_prior_obs"].eq(0).to_numpy(),
        "cold_store_segment": frame["store_segment_prior_obs"].eq(0).to_numpy(),
        "cold_title": frame["title_prior_obs"].eq(0).to_numpy(),
    }
    records = [
        metric_record(frame.loc[mask], prediction_col, dataset_name, slice_name)
        for slice_name, mask in slice_masks.items()
        if int(mask.sum()) > 0
    ]
    return pd.DataFrame.from_records(records)


def build_grouped_residual_diagnostics(
    frame: pd.DataFrame,
    prediction_col: str,
    dataset_name: str,
    group_cols: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_col in group_cols:
        for group_value, group_df in frame.groupby(group_col, dropna=False, observed=True):
            if len(group_df) < 25:
                continue
            row = metric_record(group_df, prediction_col, dataset_name, f"{group_col}={group_value}")
            row["group_col"] = group_col
            row["group_value"] = str(group_value)
            records.append(row)
    return pd.DataFrame.from_records(records)


def build_calibration_table(
    frame: pd.DataFrame,
    prediction_col: str,
    dataset_name: str,
    bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    work = frame[["sales_target", prediction_col]].copy()
    work["prediction_bin"] = pd.qcut(
        work[prediction_col].rank(method="first"),
        q=min(bins, len(work)),
        labels=False,
        duplicates="drop",
    )
    records = []
    for bin_id, bin_df in work.groupby("prediction_bin", dropna=False):
        y_true = bin_df["sales_target"].to_numpy(dtype=np.float32)
        y_pred = bin_df[prediction_col].to_numpy(dtype=np.float32)
        records.append(
            {
                "dataset": dataset_name,
                "prediction_bin": int(bin_id) if not pd.isna(bin_id) else -1,
                "rows": int(len(bin_df)),
                "actual_mean": float(np.mean(y_true)),
                "prediction_mean": float(np.mean(y_pred)),
                "bias_mean_pred_minus_actual": float(np.mean(y_pred - y_true)),
                "actual_zero_rate": float((bin_df["sales_target"] == 0).mean()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "wape": safe_wape(y_true, y_pred),
            }
        )
    return pd.DataFrame.from_records(records)


def write_explainability_outputs(
    family: str,
    booster: xgb.Booster,
    matrices: EncodedMatrices,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    shap_frames = []
    for dataset_name, frame, matrix in [
        ("validation", valid_df, matrices.valid_matrix),
        ("test", test_df, matrices.test_matrix),
    ]:
        if SHAP_MAX_ROWS <= 0 or len(frame) == 0:
            continue
        sample_size = min(SHAP_MAX_ROWS, len(frame))
        sample_positions = np.sort(
            np.random.default_rng(SEED).choice(len(frame), size=sample_size, replace=False)
        )
        sample_df = frame.iloc[sample_positions].reset_index(drop=True)
        sample_matrix = matrix[sample_positions]
        dmatrix = xgb.DMatrix(sample_matrix, feature_names=matrices.feature_names)
        shap_values = booster.predict(
            dmatrix,
            pred_contribs=True,
            iteration_range=(0, booster.best_iteration + 1),
        )
        feature_shap = shap_values[:, :-1]
        abs_feature_shap = np.abs(feature_shap)
        feature_importance = pd.DataFrame(
            {
                "family": family,
                "dataset": dataset_name,
                "feature": matrices.feature_names,
                "feature_group": [feature_group(name) for name in matrices.feature_names],
                "mean_abs_shap": abs_feature_shap.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_frames.append(feature_importance)

        group_indices: dict[str, list[int]] = defaultdict(list)
        for idx, feature_name in enumerate(matrices.feature_names):
            group_indices[feature_group(feature_name)].append(idx)

        segment_records = []
        group_values = pd.DataFrame(
            {
                group_name: abs_feature_shap[:, indices].sum(axis=1)
                for group_name, indices in group_indices.items()
            }
        )
        group_values["onsaledate"] = sample_df["onsaledate"].dt.strftime("%Y-%m-%d")
        group_values["segment"] = sample_df["segment"].astype(str)
        group_values["dataset"] = dataset_name
        for (onsaledate, segment), group_df in group_values.groupby(["onsaledate", "segment"], observed=True):
            if len(group_df) < 10:
                continue
            for group_name in group_indices:
                segment_records.append(
                    {
                        "family": family,
                        "dataset": dataset_name,
                        "onsaledate": onsaledate,
                        "segment": segment,
                        "feature_group": group_name,
                        "rows": int(len(group_df)),
                        "mean_abs_shap": float(group_df[group_name].mean()),
                    }
                )
        pd.DataFrame.from_records(segment_records).to_csv(
            OUTPUT_DIR / f"shap_feature_group_by_date_segment_{family.lower()}_{dataset_name}.csv",
            index=False,
        )

    if shap_frames:
        pd.concat(shap_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / f"shap_feature_importance_{family.lower()}.csv",
            index=False,
        )


def build_expanding_folds(family_df: pd.DataFrame) -> list[dict[str, object]]:
    if CV_FOLDS <= 0:
        return []
    cv_df = family_df.loc[family_df["split"].isin(["train", "valid"])].copy()
    dates = sorted(cv_df["onsaledate"].dt.strftime("%Y-%m-%d").unique())
    if len(dates) < CV_MIN_TRAIN_DATES + 1:
        return []
    valid_window = max(1, min(CV_VALID_DATES, max(1, len(dates) - CV_MIN_TRAIN_DATES)))
    latest_valid_start = len(dates) - valid_window
    if latest_valid_start < CV_MIN_TRAIN_DATES:
        return []
    start_candidates = np.linspace(CV_MIN_TRAIN_DATES, latest_valid_start, num=min(CV_FOLDS, latest_valid_start - CV_MIN_TRAIN_DATES + 1))
    start_indices = sorted({int(round(value)) for value in start_candidates})
    folds = []
    for fold_idx, valid_start_idx in enumerate(start_indices, start=1):
        train_dates = dates[:valid_start_idx]
        valid_dates = dates[valid_start_idx:valid_start_idx + valid_window]
        folds.append(
            {
                "fold": fold_idx,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "validation_start": valid_dates[0],
                "validation_end": valid_dates[-1],
                "train_dates": train_dates,
                "valid_dates": valid_dates,
            }
        )
    return folds


def run_rolling_validation(family: str, family_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_expanding_folds(family_df)
    if not folds:
        return pd.DataFrame(), pd.DataFrame()

    all_metrics = []
    for fold in folds:
        train_mask = family_df["onsaledate"].dt.strftime("%Y-%m-%d").isin(fold["train_dates"])
        valid_mask = family_df["onsaledate"].dt.strftime("%Y-%m-%d").isin(fold["valid_dates"])
        fold_train = family_df.loc[train_mask].copy()
        fold_valid = family_df.loc[valid_mask].copy()
        if fold_train.empty or fold_valid.empty:
            continue
        empty_test = fold_valid.iloc[0:0].copy()
        fold_train, fold_valid, _, artifact, _ = prepare_family_frames(
            family,
            fold_train,
            fold_valid,
            empty_test,
            persist=False,
        )
        matrices = encode_features(fold_train, fold_valid, empty_test, artifact)
        booster, _, dvalid = train_booster(
            family,
            fold_train,
            fold_valid,
            matrices,
            num_boost_round=CV_N_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING,
            log_label=f"rolling_fold_{fold['fold']}",
        )
        fold_valid["baseline_pred"] = compute_baseline(fold_valid)
        fold_valid["xgb_pred"] = predict_raw_units(booster, dvalid)
        baseline_metrics = build_metrics(fold_valid, "baseline_pred", "rolling_validation")
        baseline_metrics.insert(0, "model", "historical_fallback")
        xgb_metrics = build_metrics(fold_valid, "xgb_pred", "rolling_validation")
        xgb_metrics.insert(0, "model", "xgboost")
        fold_metrics = pd.concat([baseline_metrics, xgb_metrics], ignore_index=True)
        fold_metrics.insert(0, "family", family)
        for key, value in fold.items():
            if key not in {"train_dates", "valid_dates"}:
                fold_metrics[key] = value
        fold_metrics["best_iteration"] = int(booster.best_iteration)
        fold_metrics["early_stopping_best_score"] = float(booster.best_score)
        all_metrics.append(fold_metrics)

    if not all_metrics:
        return pd.DataFrame(), pd.DataFrame()

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = (
        metrics_df.groupby(["family", "model", "slice"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_wape=("wape", "mean"),
            std_wape=("wape", "std"),
        )
    )
    return metrics_df, summary_df


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

    family_all_df = pd.concat([family_train_df, family_valid_df, family_test_df], ignore_index=True)
    rolling_metrics_df, rolling_summary_df = run_rolling_validation(family, family_all_df)

    family_train_df, family_valid_df, family_test_df, artifact, preprocessing_diagnostics = prepare_family_frames(
        family,
        family_train_df,
        family_valid_df,
        family_test_df,
        persist=True,
    )
    matrices = encode_features(family_train_df, family_valid_df, family_test_df, artifact)

    test_labels = np.log1p(family_test_df["sales_target"].to_numpy(dtype=np.float32))

    test_weights = build_sample_weights(family_test_df)

    booster, dtrain, dvalid = train_booster(
        family,
        family_train_df,
        family_valid_df,
        matrices,
        num_boost_round=N_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING,
        log_label="final",
    )
    dtest = xgb.DMatrix(matrices.test_matrix, label=test_labels, weight=test_weights, feature_names=matrices.feature_names)

    family_valid_df["baseline_pred"] = compute_baseline(family_valid_df)
    family_test_df["baseline_pred"] = compute_baseline(family_test_df)
    family_valid_df["xgb_pred"] = predict_raw_units(booster, dvalid)
    family_test_df["xgb_pred"] = predict_raw_units(booster, dtest)

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

    diagnostics_df = pd.concat(
        [
            build_named_slice_diagnostics(family_valid_df, "xgb_pred", "validation"),
            build_named_slice_diagnostics(family_test_df, "xgb_pred", "test"),
            build_grouped_residual_diagnostics(
                family_valid_df,
                "xgb_pred",
                "validation",
                ["onsaledate", "segment", "subsegment", "store_chain", "classoftrade"],
            ),
            build_grouped_residual_diagnostics(
                family_test_df,
                "xgb_pred",
                "test",
                ["onsaledate", "segment", "subsegment", "store_chain", "classoftrade"],
            ),
        ],
        ignore_index=True,
    )
    diagnostics_df.insert(0, "family", family)
    diagnostics_df.to_csv(OUTPUT_DIR / f"residual_diagnostics_{family.lower()}.csv", index=False)

    calibration_df = pd.concat(
        [
            build_calibration_table(family_valid_df, "xgb_pred", "validation"),
            build_calibration_table(family_test_df, "xgb_pred", "test"),
        ],
        ignore_index=True,
    )
    calibration_df.insert(0, "family", family)
    calibration_df.to_csv(OUTPUT_DIR / f"calibration_{family.lower()}.csv", index=False)

    write_explainability_outputs(family, booster, matrices, family_valid_df, family_test_df)

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
        "preprocessing_fit_scope": "training_split_only",
        "preprocessing_unknown_category_rates": {
            dataset_name: diagnostics["unknown_category_rates"]
            for dataset_name, diagnostics in preprocessing_diagnostics.items()
        },
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
        rolling_metrics_df=rolling_metrics_df.assign(
            early_stopping_metric="rmse_on_log1p_target",
            prediction_metric_scale="raw_copy_counts",
        ) if not rolling_metrics_df.empty else rolling_metrics_df,
        rolling_summary_df=rolling_summary_df,
        test_predictions=test_predictions,
        importance_df=importance_df,
        grouped_importance_df=grouped_importance_df,
    )


def main() -> int:
    if CORE_FEATURES_CSV:
        core_df = normalize_core_dataframe(load_existing_dataframe(CORE_FEATURES_CSV))
        split_info = summarize_existing_split(core_df)
    elif DEFAULT_CORE_FEATURES_CACHE.exists():
        core_df = normalize_core_dataframe(load_existing_dataframe(str(DEFAULT_CORE_FEATURES_CACHE)))
        split_info = summarize_existing_split(core_df)
    else:
        split_info = compute_split_dates(TRAIN_SPLIT_RATIO, VALID_SPLIT_RATIO, TEST_SPLIT_RATIO)
        core_query = build_core_feature_query(
            validation_start=split_info["validation_start"],
            test_start=split_info["test_start"],
            train_sample_pct=TRAIN_SAMPLE_PCT,
        )
        core_df = normalize_core_dataframe(export_query_to_dataframe(core_query, "core_features", DEFAULT_CORE_FEATURES_CACHE))

    log(
        "Using chronological split: "
        f"train {split_info['train_start']} to {split_info['train_end']} "
        f"({split_info['train_dates']} dates), "
        f"validation {split_info['validation_start']} to {split_info['validation_end']} "
        f"({split_info['valid_dates']} dates), "
        f"test {split_info['test_start']} to {split_info['test_end']} "
        f"({split_info['test_dates']} dates)"
    )

    train_df, valid_df, test_df, modeling_filter_counts = split_modeling_frames(core_df)
    family_results = []
    for family in FAMILIES:
        family_results.append(train_family_model(family, train_df, valid_df, test_df))

    metrics_df = pd.concat([result.metrics_df for result in family_results], ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    rolling_metrics = [result.rolling_metrics_df for result in family_results if not result.rolling_metrics_df.empty]
    if rolling_metrics:
        rolling_metrics_df = pd.concat(rolling_metrics, ignore_index=True)
        rolling_metrics_df.to_csv(OUTPUT_DIR / "rolling_validation_metrics.csv", index=False)

    rolling_summaries = [result.rolling_summary_df for result in family_results if not result.rolling_summary_df.empty]
    if rolling_summaries:
        rolling_summary_df = pd.concat(rolling_summaries, ignore_index=True)
        rolling_summary_df.to_csv(OUTPUT_DIR / "rolling_validation_summary.csv", index=False)
    else:
        rolling_summary_df = pd.DataFrame()

    test_output = pd.concat([result.test_predictions for result in family_results], ignore_index=True)
    test_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    summary = {
        "db_uri": DB_URI,
        "train_sample_pct": TRAIN_SAMPLE_PCT,
        "train_ratio": TRAIN_SPLIT_RATIO,
        "validation_ratio": VALID_SPLIT_RATIO,
        "test_ratio": TEST_SPLIT_RATIO,
        "cv_folds": CV_FOLDS,
        "cv_valid_dates": CV_VALID_DATES,
        "cv_min_train_dates": CV_MIN_TRAIN_DATES,
        "cv_n_rounds": CV_N_ROUNDS,
        "shap_max_rows": SHAP_MAX_ROWS,
        "train_start": split_info["train_start"],
        "train_end": split_info["train_end"],
        "validation_start": split_info["validation_start"],
        "validation_end": split_info["validation_end"],
        "test_start": split_info["test_start"],
        "test_end": split_info["test_end"],
        "modeling_filter_counts": modeling_filter_counts,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "family_runs": {
            result.family: {
                "train_rows": int((train_df["type"] == result.family).sum()),
                "valid_rows": int((valid_df["type"] == result.family).sum()),
                "test_rows": int((test_df["type"] == result.family).sum()),
                "zero_sales_rows_kept": {
                    "train": int(((train_df["type"] == result.family) & (train_df["sales_target"] == 0)).sum()),
                    "validation": int(((valid_df["type"] == result.family) & (valid_df["sales_target"] == 0)).sum()),
                    "test": int(((test_df["type"] == result.family) & (test_df["sales_target"] == 0)).sum()),
                },
                "test_metrics": result.test_metrics,
            }
            for result in family_results
        },
        "training_diagnostics": {
            result.family: result.training_diagnostics
            for result in family_results
        },
        "rolling_validation_summary": rolling_summary_df.to_dict(orient="records") if not rolling_summary_df.empty else [],
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"Finished. Outputs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
