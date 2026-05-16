#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parent.parent
NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "modeling_datasets"
DEFAULT_DB_URI = os.getenv("PAPERRUSH_DB_URI", "postgresql://norbert.jaworski@/paperrush?host=/tmp")
FAMILIES = ("Weeklies", "SIP")
EMBEDDING_PCA_COMPONENTS = 130
EMBEDDING_FEATURES = [f"embedding_pca_{idx:03d}" for idx in range(1, EMBEDDING_PCA_COMPONENTS + 1)]
EMBEDDING_ANALOG_K = 10
EMBEDDING_ANALOG_PRIOR_STRENGTH = 5.0
EMBEDDING_ANALOG_FEATURES = [
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_neighbor_count",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_effective_neighbor_count",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_top1_similarity",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_mean_similarity",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weight_sum",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_avg_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_positive_sale_rate",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_p50_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_p75_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_p90_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_p90_minus_p50_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_weighted_p90_over_avg_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_subsegment_prior_positive_rate",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_subsegment_prior_avg_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_subsegment_prior_p50_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_subsegment_prior_p75_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_subsegment_prior_p90_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_positive_sale_rate",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_avg_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p50_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p75_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p90_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p90_minus_p50_positive_sales",
    f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p90_over_avg_positive_sales",
]

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

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

COMMON_CATEGORICAL_FEATURES = [
    "store_id",
    "onsale_month_cat",
    "segment",
    "subsegment",
    "frequency",
    "store_chain",
    "region",
    "classoftrade",
]

FAMILY_CATEGORICAL_FEATURES = {
    "Weeklies": ["title", *COMMON_CATEGORICAL_FEATURES],
    "SIP": COMMON_CATEGORICAL_FEATURES,
}

COMMON_NUMERIC_FEATURES = [
    "price",
    "issue_length_days",
    "onsale_month",
    "onsale_week",
    "onsale_dow",
    "merchandised",
    "facings",
    "pockets",
    "global_prior_positive_rate",
    "global_prior_positive_avg_sales",
    "global_prior_obs",
    "store_prior_positive_rate",
    "store_prior_positive_avg_sales",
    "store_prior_obs",
    "store_title_prior_positive_rate",
    "store_title_prior_positive_avg_sales",
    "store_title_prior_obs",
    "store_segment_prior_positive_rate",
    "store_segment_prior_positive_avg_sales",
    "store_segment_prior_obs",
    "store_subsegment_prior_positive_rate",
    "store_subsegment_prior_positive_avg_sales",
    "store_subsegment_prior_obs",
    "store_type_prior_positive_rate",
    "store_type_prior_positive_avg_sales",
    "store_type_prior_obs",
    "chain_segment_prior_positive_rate",
    "chain_segment_prior_positive_avg_sales",
    "chain_segment_prior_obs",
    "title_prior_positive_rate",
    "title_prior_positive_avg_sales",
    "title_prior_obs",
    "segment_prior_positive_rate",
    "segment_prior_positive_avg_sales",
    "segment_prior_obs",
    *EMBEDDING_ANALOG_FEATURES,
    *EMBEDDING_FEATURES,
    *BASE_DEMOGRAPHIC_COLUMNS,
]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Parquet output requires pyarrow. Install it with: python3 -m pip install pyarrow"
        ) from exc


def run_psql_copy(db_uri: str, query: str, destination: Path) -> None:
    copy_cmd = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
    cmd = ["psql", db_uri, "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-c", copy_cmd]
    with destination.open("wb") as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def run_psql_csv(db_uri: str, query: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="paperrush_embeddings_") as tmp_dir:
        csv_path = Path(tmp_dir) / "query.csv"
        run_psql_copy(db_uri, query, csv_path)
        return pd.read_csv(csv_path)


def parse_embedding(value: str) -> np.ndarray:
    return np.fromstring(str(value).strip("[]"), sep=",", dtype=np.float64)


def load_family_embeddings(db_uri: str, family: str) -> tuple[pd.DataFrame, np.ndarray]:
    query = f"""
    select
        dp.product_id,
        dp.onsaledate,
        coalesce(dp.offsaledate, dp.onsaledate) as offsaledate,
        {category_sql("dp.subsegment")} as subsegment,
        count(fs.*) filter (where fs.drawqty <> 0) as observed_rows,
        count(fs.*) filter (where fs.drawqty <> 0 and fs.soldqty > 0) as positive_rows,
        coalesce(sum(fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as positive_sales_sum,
        coalesce(avg(fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as avg_positive_sales,
        coalesce(percentile_cont(0.50) within group (order by fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as p50_positive_sales,
        coalesce(percentile_cont(0.75) within group (order by fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as p75_positive_sales,
        coalesce(percentile_cont(0.90) within group (order by fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as p90_positive_sales,
        ce.embedding::text as embedding
    from core.dim_product dp
    join core.content_embedding ce on ce.product_id = dp.product_id
    left join core.fact_sale fs on fs.product_id = dp.product_id
    where dp.type = '{family}'
    group by dp.product_id, dp.onsaledate, coalesce(dp.offsaledate, dp.onsaledate), {category_sql("dp.subsegment")}, ce.embedding
    order by dp.onsaledate, dp.product_id
    """
    embedding_frame = run_psql_csv(db_uri, query)
    if embedding_frame.empty:
        raise ValueError(f"No content embeddings found for {family}")
    matrix = np.vstack(embedding_frame["embedding"].map(parse_embedding).to_numpy())
    if matrix.shape[1] < EMBEDDING_PCA_COMPONENTS:
        raise ValueError(
            f"{family} embeddings have only {matrix.shape[1]} dimensions; "
            f"need at least {EMBEDDING_PCA_COMPONENTS}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{family} embeddings contain non-finite values")
    embedding_frame = embedding_frame.drop(columns=["embedding"])
    embedding_frame["onsaledate"] = pd.to_datetime(embedding_frame["onsaledate"])
    embedding_frame["offsaledate"] = pd.to_datetime(embedding_frame["offsaledate"])
    embedding_frame["positive_sale_rate"] = (
        embedding_frame["positive_rows"].astype(float) / embedding_frame["observed_rows"].replace(0, np.nan)
    ).fillna(0.0)
    return embedding_frame.copy(), matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def effective_neighbor_count(weights: np.ndarray) -> float:
    weight_square_sum = float(np.square(weights).sum())
    if weight_square_sum <= 0:
        return 0.0
    return float(np.square(weights.sum()) / weight_square_sum)


def aggregate_prior_stats(
    candidate_indices: np.ndarray,
    positive_sales_sum: np.ndarray,
    positive_rows: np.ndarray,
    observed_rows: np.ndarray,
    quantile_stats: np.ndarray,
) -> np.ndarray:
    if len(candidate_indices) == 0:
        return np.zeros(5, dtype=np.float64)

    candidate_positive_rows = positive_rows[candidate_indices].astype(np.float64)
    candidate_observed_rows = observed_rows[candidate_indices].astype(np.float64)
    positive_row_sum = float(candidate_positive_rows.sum())
    observed_row_sum = float(candidate_observed_rows.sum())
    positive_rate = safe_ratio(positive_row_sum, observed_row_sum)
    avg_positive_sales = safe_ratio(float(positive_sales_sum[candidate_indices].sum()), positive_row_sum)

    positive_mask = candidate_positive_rows > 0
    if positive_mask.any():
        quantiles = np.average(
            quantile_stats[candidate_indices][positive_mask],
            axis=0,
            weights=candidate_positive_rows[positive_mask],
        )
    else:
        quantiles = np.zeros(3, dtype=np.float64)

    return np.array(
        [
            positive_rate,
            avg_positive_sales,
            float(quantiles[0]),
            float(quantiles[1]),
            float(quantiles[2]),
        ],
        dtype=np.float64,
    )


def shrink_stats(raw_stats: np.ndarray, prior_stats: np.ndarray, effective_count: float) -> np.ndarray:
    strength = EMBEDDING_ANALOG_PRIOR_STRENGTH
    return (effective_count * raw_stats + strength * prior_stats) / (effective_count + strength)


def build_embedding_analog_features(embedding_products: pd.DataFrame, embedding_matrix: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    product_ids = embedding_products["product_id"].to_numpy(dtype=np.int64)
    onsaledates = embedding_products["onsaledate"].to_numpy(dtype="datetime64[ns]")
    offsaledates = embedding_products["offsaledate"].to_numpy(dtype="datetime64[ns]")
    subsegments = embedding_products["subsegment"].astype(str).to_numpy()
    observed_rows = embedding_products["observed_rows"].to_numpy(dtype=np.float64)
    positive_rows = embedding_products["positive_rows"].to_numpy(dtype=np.float64)
    positive_sales_sum = embedding_products["positive_sales_sum"].to_numpy(dtype=np.float64)
    normalized_embeddings = row_normalize(embedding_matrix)
    stats = embedding_products[
        [
            "avg_positive_sales",
            "positive_sale_rate",
            "p50_positive_sales",
            "p75_positive_sales",
            "p90_positive_sales",
        ]
    ].to_numpy(dtype=np.float64)
    quantile_stats = embedding_products[
        [
            "p50_positive_sales",
            "p75_positive_sales",
            "p90_positive_sales",
        ]
    ].to_numpy(dtype=np.float64)
    result = np.zeros((len(embedding_products), len(EMBEDDING_ANALOG_FEATURES)), dtype=np.float32)
    sorted_indices = np.argsort(onsaledates, kind="stable")

    for position, product_idx in enumerate(sorted_indices):
        current_date = onsaledates[product_idx]
        candidate_indices = sorted_indices[:position]
        if len(candidate_indices):
            candidate_indices = candidate_indices[
                (onsaledates[candidate_indices] < current_date)
                & (offsaledates[candidate_indices] < current_date)
            ]
        if len(candidate_indices) == 0:
            continue

        similarities = normalized_embeddings[candidate_indices] @ normalized_embeddings[product_idx]
        k = min(EMBEDDING_ANALOG_K, len(candidate_indices))
        if k < len(candidate_indices):
            top_positions = np.argpartition(similarities, -k)[-k:]
            top_positions = top_positions[np.argsort(similarities[top_positions])[::-1]]
        else:
            top_positions = np.argsort(similarities)[::-1]
        neighbor_indices = candidate_indices[top_positions]
        neighbor_similarities = similarities[top_positions]
        weights = np.clip(neighbor_similarities, 0.0, None)
        if float(weights.sum()) == 0.0:
            weights = np.ones_like(neighbor_similarities, dtype=np.float64)
        neighbor_stats = stats[neighbor_indices]
        weighted_stats = np.average(neighbor_stats, axis=0, weights=weights)
        effective_count = effective_neighbor_count(weights)
        current_subsegment = subsegments[product_idx]
        subsegment_candidate_indices = candidate_indices[subsegments[candidate_indices] == current_subsegment]
        prior_indices = subsegment_candidate_indices if len(subsegment_candidate_indices) else candidate_indices
        subsegment_prior_stats = aggregate_prior_stats(
            prior_indices,
            positive_sales_sum=positive_sales_sum,
            positive_rows=positive_rows,
            observed_rows=observed_rows,
            quantile_stats=quantile_stats,
        )
        # Align raw stats as rate, avg, p50, p75, p90 for shrinkage.
        raw_stats = np.array(
            [
                float(weighted_stats[1]),
                float(weighted_stats[0]),
                float(weighted_stats[2]),
                float(weighted_stats[3]),
                float(weighted_stats[4]),
            ],
            dtype=np.float64,
        )
        shrunk = shrink_stats(raw_stats, subsegment_prior_stats, effective_count)
        raw_p90_minus_p50 = float(weighted_stats[4] - weighted_stats[2])
        raw_p90_over_avg = safe_ratio(float(weighted_stats[4]), float(weighted_stats[0]))
        shrunk_p90_minus_p50 = float(shrunk[4] - shrunk[2])
        shrunk_p90_over_avg = safe_ratio(float(shrunk[4]), float(shrunk[1]))
        result[product_idx] = np.array(
            [
                float(k),
                effective_count,
                float(neighbor_similarities[0]),
                float(neighbor_similarities.mean()),
                float(weights.sum()),
                float(weighted_stats[0]),
                float(weighted_stats[1]),
                float(weighted_stats[2]),
                float(weighted_stats[3]),
                float(weighted_stats[4]),
                raw_p90_minus_p50,
                raw_p90_over_avg,
                float(subsegment_prior_stats[0]),
                float(subsegment_prior_stats[1]),
                float(subsegment_prior_stats[2]),
                float(subsegment_prior_stats[3]),
                float(subsegment_prior_stats[4]),
                float(shrunk[0]),
                float(shrunk[1]),
                float(shrunk[2]),
                float(shrunk[3]),
                float(shrunk[4]),
                shrunk_p90_minus_p50,
                shrunk_p90_over_avg,
            ],
            dtype=np.float32,
        )

    analog_frame = pd.DataFrame({"product_id": product_ids})
    for idx, feature in enumerate(EMBEDDING_ANALOG_FEATURES):
        analog_frame[feature] = result[:, idx]
    summary = {
        "method": "same_family_prior_product_cosine_neighbors",
        "neighbor_k": EMBEDDING_ANALOG_K,
        "feature_names": EMBEDDING_ANALOG_FEATURES,
        "products": int(len(embedding_products)),
        "products_with_prior_neighbors": int((result[:, 0] > 0).sum()),
        "products_without_prior_neighbors": int((result[:, 0] == 0).sum()),
        "strictly_prior_onsaledate": True,
        "candidate_completion_rule": "neighbor offsaledate must be before current product onsaledate",
        "bayesian_shrinkage_prior": "same-subsegment completed-prior aggregate, falling back to same-family completed-prior aggregate",
        "bayesian_shrinkage_prior_strength": EMBEDDING_ANALOG_PRIOR_STRENGTH,
        "effective_neighbor_count": "Kish effective sample size: sum(weights)^2 / sum(weights^2)",
        "similarity_weighting": "cosine weights clipped at zero; equal weights fallback when all selected similarities are non-positive",
    }
    return analog_frame, summary


def add_embedding_features(frame: pd.DataFrame, family: str, db_uri: str, family_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    log(f"Loading {family} content embeddings for PCA and analog features")
    embedding_products, embedding_matrix = load_family_embeddings(db_uri, family)
    product_ids = embedding_products["product_id"].to_numpy()
    product_id_to_index = {int(product_id): idx for idx, product_id in enumerate(product_ids)}
    train_product_ids = pd.unique(frame.loc[frame["split"] == "train", "product_id"])
    train_indices = [product_id_to_index[int(product_id)] for product_id in train_product_ids if int(product_id) in product_id_to_index]
    if len(train_indices) < EMBEDDING_PCA_COMPONENTS:
        raise ValueError(
            f"Only {len(train_indices)} training products with embeddings for {family}; "
            f"need at least {EMBEDDING_PCA_COMPONENTS}"
        )

    log(f"Fitting {family} PCA reducer on {len(train_indices)} train products")
    pca = PCA(n_components=EMBEDDING_PCA_COMPONENTS, svd_solver="full", random_state=42)
    pca.fit(embedding_matrix[train_indices])
    reduced_products = pca.transform(embedding_matrix).astype(np.float32)

    product_indices = frame["product_id"].map(product_id_to_index).to_numpy()
    missing_mask = pd.isna(product_indices)
    row_reduced = np.zeros((len(frame), EMBEDDING_PCA_COMPONENTS), dtype=np.float32)
    if (~missing_mask).any():
        row_reduced[~missing_mask] = reduced_products[product_indices[~missing_mask].astype(np.int64)]

    embedding_feature_frame = pd.DataFrame(row_reduced, columns=EMBEDDING_FEATURES, index=frame.index)
    frame = pd.concat([frame, embedding_feature_frame], axis=1)
    log(f"Building {family} leakage-safe embedding analog features")
    analog_product_frame, analog_summary = build_embedding_analog_features(embedding_products, embedding_matrix)
    frame = frame.merge(analog_product_frame, on="product_id", how="left")
    frame[EMBEDDING_ANALOG_FEATURES] = frame[EMBEDDING_ANALOG_FEATURES].fillna(0.0).astype(np.float32)

    artifact = {
        "family": family,
        "method": "pca",
        "fit_scope": "training_split_products_only",
        "raw_embedding_dimensions": int(embedding_matrix.shape[1]),
        "requested_components": EMBEDDING_PCA_COMPONENTS,
        "actual_components": int(pca.n_components_),
        "feature_names": EMBEDDING_FEATURES,
        "train_products_with_embeddings": int(len(train_indices)),
        "family_products_with_embeddings": int(len(product_ids)),
        "row_coverage": {
            "rows": int(len(frame)),
            "rows_with_embedding": int((~missing_mask).sum()),
            "rows_missing_embedding": int(missing_mask.sum()),
            "row_coverage_rate": float((~missing_mask).mean()),
        },
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "cumulative_explained_variance": [float(value) for value in np.cumsum(pca.explained_variance_ratio_)],
        "mean": [float(value) for value in pca.mean_],
        "components": pca.components_.astype(float).tolist(),
        "analog_features": analog_summary,
    }
    artifact_path = family_dir / "embedding_pca_regressor_positive.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return frame, {
        key: value
        for key, value in artifact.items()
        if key not in {"mean", "components", "explained_variance_ratio", "cumulative_explained_variance"}
    } | {
        "artifact_path": str(artifact_path),
        "cumulative_explained_variance_at_130": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "analog_features": analog_summary,
    }


def category_sql(column: str) -> str:
    return f"coalesce(nullif(trim({column}), ''), 'UNKNOWN')"


def history_cte(name: str, group_cols: list[str]) -> str:
    group_expr = ", ".join(["onsaledate", *group_cols])
    partition_expr = ", ".join(group_cols)
    select_group_cols = ",\n        ".join(group_cols)
    if select_group_cols:
        select_group_cols = f",\n        {select_group_cols}"
    return f"""
{name}_date as (
    select
        {group_expr},
        sum(sales_target) filter (where positive_sale_flag = 1) as positive_sales_sum,
        sum(positive_sale_flag) as positive_row_count,
        count(*) as row_count
    from base_all
    group by {group_expr}
),
{name}_hist as (
    select
        onsaledate
        {select_group_cols},
        sum(positive_sales_sum) over (
            partition by {partition_expr}
            order by onsaledate rows between unbounded preceding and 1 preceding
        ) as prior_positive_sales_sum,
        sum(positive_row_count) over (
            partition by {partition_expr}
            order by onsaledate rows between unbounded preceding and 1 preceding
        ) as prior_positive_row_count,
        sum(row_count) over (
            partition by {partition_expr}
            order by onsaledate rows between unbounded preceding and 1 preceding
        ) as prior_row_count
    from {name}_date
)"""


def global_history_cte() -> str:
    return """
global_date as (
    select
        onsaledate,
        sum(sales_target) filter (where positive_sale_flag = 1) as positive_sales_sum,
        sum(positive_sale_flag) as positive_row_count,
        count(*) as row_count
    from base_all
    group by onsaledate
),
global_hist as (
    select
        onsaledate,
        sum(positive_sales_sum) over (order by onsaledate rows between unbounded preceding and 1 preceding) as prior_positive_sales_sum,
        sum(positive_row_count) over (order by onsaledate rows between unbounded preceding and 1 preceding) as prior_positive_row_count,
        sum(row_count) over (order by onsaledate rows between unbounded preceding and 1 preceding) as prior_row_count
    from global_date
)"""


def prior_rate(prefix: str, alias: str) -> str:
    return f"""
    coalesce({alias}.prior_positive_row_count::double precision / nullif({alias}.prior_row_count, 0), 0) as {prefix}_prior_positive_rate,
    coalesce({alias}.prior_positive_sales_sum / nullif({alias}.prior_positive_row_count, 0), 0) as {prefix}_prior_positive_avg_sales,
    coalesce({alias}.prior_row_count, 0) as {prefix}_prior_obs"""


def build_regressor_positive_query(family: str) -> str:
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
),
{global_history_cte()},
{history_cte("store", ["store_id"])},
{history_cte("store_title", ["store_id", "title"])},
{history_cte("store_segment", ["store_id", "segment"])},
{history_cte("store_subsegment", ["store_id", "subsegment"])},
{history_cte("store_type", ["store_id", "type"])},
{history_cte("chain_segment", ["store_chain", "segment"])},
{history_cte("title", ["title"])},
{history_cte("segment", ["segment"])},
target as (
    select *
    from base_all
    where positive_sale_flag = 1
)
select
    t.store_id,
    t.product_id,
    t.onsaledate,
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
{prior_rate("global", "gh")},
{prior_rate("store", "sh")},
{prior_rate("store_title", "sth")},
{prior_rate("store_segment", "sgh")},
{prior_rate("store_subsegment", "ssh")},
{prior_rate("store_type", "styh")},
{prior_rate("chain_segment", "csh")},
{prior_rate("title", "th")},
{prior_rate("segment", "segh")},
    {demo_select}
from target t
left join global_hist gh on gh.onsaledate = t.onsaledate
left join store_hist sh on sh.onsaledate = t.onsaledate and sh.store_id = t.store_id
left join store_title_hist sth on sth.onsaledate = t.onsaledate and sth.store_id = t.store_id and sth.title = t.title
left join store_segment_hist sgh on sgh.onsaledate = t.onsaledate and sgh.store_id = t.store_id and sgh.segment = t.segment
left join store_subsegment_hist ssh on ssh.onsaledate = t.onsaledate and ssh.store_id = t.store_id and ssh.subsegment = t.subsegment
left join store_type_hist styh on styh.onsaledate = t.onsaledate and styh.store_id = t.store_id and styh.type = t.type
left join chain_segment_hist csh on csh.onsaledate = t.onsaledate and csh.store_chain = t.store_chain and csh.segment = t.segment
left join title_hist th on th.onsaledate = t.onsaledate and th.title = t.title
left join segment_hist segh on segh.onsaledate = t.onsaledate and segh.segment = t.segment
order by t.onsaledate, t.product_id, t.store_id
"""


def split_dates(dates: list[str], train_ratio: float, valid_ratio: float, test_ratio: float) -> dict[str, object]:
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    n_dates = len(dates)
    if n_dates < 3:
        raise ValueError("Need at least three dates for train/validation/test splits")
    train_n = max(1, round(n_dates * train_ratio))
    valid_n = max(1, round(n_dates * valid_ratio))
    if train_n + valid_n >= n_dates:
        valid_n = max(1, n_dates - train_n - 1)
    test_n = n_dates - train_n - valid_n
    if test_n < 1:
        test_n = 1
        if valid_n > 1:
            valid_n -= 1
        else:
            train_n -= 1
    return {
        "train_dates": dates[:train_n],
        "validation_dates": dates[train_n:train_n + valid_n],
        "test_dates": dates[train_n + valid_n:],
    }


def build_expanding_folds(
    train_valid_dates: list[str],
    folds: int,
    valid_dates_per_fold: int,
    min_train_dates: int,
) -> list[dict[str, object]]:
    if folds <= 0:
        return []
    valid_window = max(1, valid_dates_per_fold)
    latest_start = len(train_valid_dates) - valid_window
    if latest_start < min_train_dates:
        return []
    raw_starts = np.linspace(min_train_dates, latest_start, num=min(folds, latest_start - min_train_dates + 1))
    starts = sorted({int(round(value)) for value in raw_starts})
    result = []
    for idx, start in enumerate(starts, start=1):
        train_dates = train_valid_dates[:start]
        valid_dates = train_valid_dates[start:start + valid_window]
        result.append(
            {
                "fold": idx,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "validation_start": valid_dates[0],
                "validation_end": valid_dates[-1],
                "train_dates": train_dates,
                "validation_dates": valid_dates,
            }
        )
    return result


def assign_split(frame: pd.DataFrame, split_info: dict[str, list[str]]) -> pd.DataFrame:
    date_to_split = {}
    for split_name, dates in split_info.items():
        clean_name = "valid" if split_name == "validation_dates" else split_name.replace("_dates", "")
        date_to_split.update({date: clean_name for date in dates})
    frame = frame.copy()
    frame["onsaledate"] = pd.to_datetime(frame["onsaledate"])
    frame["split"] = frame["onsaledate"].dt.strftime("%Y-%m-%d").map(date_to_split)
    if frame["split"].isna().any():
        raise ValueError("Some rows did not map to a train/valid/test split")
    return frame


def feature_contract(family: str) -> dict[str, object]:
    return {
        "family": family,
        "stage": "regressor_positive",
        "row_filter": "DrawQty != 0 and SoldQty > 0",
        "target": "sales_target",
        "target_transform_candidates": [
            "log1p_squarederror",
            "count_poisson",
            "tweedie",
            "quantile_50",
            "quantile_80",
            "quantile_90",
        ],
        "id_columns": [
            "store_id",
            "product_id",
            "onsaledate",
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


def build_family_dataset(
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
    query = build_regressor_positive_query(family)
    with tempfile.TemporaryDirectory(prefix="paperrush_modeling_") as tmp_dir:
        csv_path = Path(tmp_dir) / f"{family.lower()}_regressor_positive.csv"
        log(f"Exporting {family} positive-sales regressor dataset from Postgres")
        run_psql_copy(db_uri, query, csv_path)
        log(f"Loading {family} CSV export")
        frame = pd.read_csv(csv_path, parse_dates=["onsaledate"])

    dates = sorted(frame["onsaledate"].dt.strftime("%Y-%m-%d").unique())
    split_info = split_dates(dates, train_ratio, valid_ratio, test_ratio)
    frame = assign_split(frame, split_info)
    family_dir = output_dir / family.lower()
    family_dir.mkdir(parents=True, exist_ok=True)
    frame, embedding_summary = add_embedding_features(frame, family, db_uri, family_dir)
    parquet_path = family_dir / "regressor_positive.parquet"
    frame.to_parquet(parquet_path, index=False)

    contract = feature_contract(family)
    contract_path = family_dir / "feature_contract_regressor_positive.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    folds_payload = build_expanding_folds(
        split_info["train_dates"] + split_info["validation_dates"],
        folds=folds,
        valid_dates_per_fold=valid_dates_per_fold,
        min_train_dates=min_train_dates,
    )
    summary = {
        "family": family,
        "stage": "regressor_positive",
        "parquet_path": str(parquet_path),
        "feature_contract_path": str(contract_path),
        "rows": int(len(frame)),
        "dates": len(dates),
        "train_rows": int((frame["split"] == "train").sum()),
        "valid_rows": int((frame["split"] == "valid").sum()),
        "test_rows": int((frame["split"] == "test").sum()),
        "sales_target_summary": {
            "mean": float(frame["sales_target"].mean()),
            "median": float(frame["sales_target"].median()),
            "p90": float(frame["sales_target"].quantile(0.90)),
            "p99": float(frame["sales_target"].quantile(0.99)),
        },
        "embedding_features": embedding_summary,
        "split_dates": split_info,
        "folds": folds_payload,
    }
    (family_dir / "manifest_regressor_positive.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build positive-sales regression modeling datasets.")
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
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
        build_family_dataset(
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
    existing_by_family: dict[str, object] = {}
    manifest_path = args.output_dir / "split_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_by_family = {
            item["family"]: item
            for item in existing.get("families", [])
        }
    for summary in summaries:
        existing_by_family[summary["family"]] = summary
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "db_uri": args.db_uri,
        "families": [existing_by_family[family] for family in sorted(existing_by_family)],
    }
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    log(f"Wrote modeling datasets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
