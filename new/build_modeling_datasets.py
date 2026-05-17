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
EMBEDDING_AFFINITY_COMPONENTS = 32
EMBEDDING_AFFINITY_LOW_QUANTILE = 0.30
EMBEDDING_AFFINITY_TAIL_QUANTILE = 0.80
EMBEDDING_AFFINITY_CONTEXTS: dict[str, list[str]] = {
    "global_affinity": [],
    "store_affinity": ["store_id"],
    "store_segment_affinity": ["store_id", "segment"],
    "store_subsegment_affinity": ["store_id", "subsegment"],
    "store_title_affinity": ["store_id", "title"],
    "chain_title_affinity": ["store_chain", "title"],
    "class_title_affinity": ["classoftrade", "title"],
    "chain_segment_affinity": ["store_chain", "segment"],
    "chain_class_subsegment_affinity": ["store_chain", "classoftrade", "subsegment"],
    "class_subsegment_affinity": ["classoftrade", "subsegment"],
    "chain_affinity": ["store_chain"],
    "class_affinity": ["classoftrade"],
    "segment_affinity": ["segment"],
    "subsegment_affinity": ["subsegment"],
}


def embedding_affinity_feature_names() -> list[str]:
    names = []
    for prefix in EMBEDDING_AFFINITY_CONTEXTS:
        names.extend(
            [
                f"{prefix}_tail_similarity",
                f"{prefix}_low_similarity",
                f"{prefix}_tail_minus_low_similarity",
                f"{prefix}_tail_obs",
                f"{prefix}_low_obs",
                f"{prefix}_tail_avg_sales",
                f"{prefix}_low_avg_sales",
            ]
        )
    return names


EMBEDDING_AFFINITY_PRIOR_CONTEXTS: dict[str, tuple[str, str]] = {
    "global_affinity": ("global_prior_positive_avg_sales", "global_prior_obs"),
    "store_affinity": ("store_prior_positive_avg_sales", "store_prior_obs"),
    "store_segment_affinity": ("store_segment_prior_positive_avg_sales", "store_segment_prior_obs"),
    "store_subsegment_affinity": ("store_subsegment_prior_positive_avg_sales", "store_subsegment_prior_obs"),
    "store_title_affinity": ("store_title_prior_positive_avg_sales", "store_title_prior_obs"),
    "chain_title_affinity": ("chain_title_prior_positive_avg_sales", "chain_title_prior_obs"),
    "class_title_affinity": ("class_title_prior_positive_avg_sales", "class_title_prior_obs"),
    "chain_segment_affinity": ("chain_segment_prior_positive_avg_sales", "chain_segment_prior_obs"),
    "chain_class_subsegment_affinity": (
        "chain_class_subsegment_prior_positive_avg_sales",
        "chain_class_subsegment_prior_obs",
    ),
    "class_subsegment_affinity": ("class_subsegment_prior_positive_avg_sales", "class_subsegment_prior_obs"),
    "chain_affinity": ("chain_prior_positive_avg_sales", "chain_prior_obs"),
    "class_affinity": ("class_prior_positive_avg_sales", "class_prior_obs"),
    "segment_affinity": ("segment_prior_positive_avg_sales", "segment_prior_obs"),
    "subsegment_affinity": ("subsegment_prior_positive_avg_sales", "subsegment_prior_obs"),
}


def embedding_affinity_derived_feature_names() -> list[str]:
    names = []
    for prefix in EMBEDDING_AFFINITY_PRIOR_CONTEXTS:
        names.extend(
            [
                f"{prefix}_tail_signal_x_analog_p90",
                f"{prefix}_tail_signal_x_prior_avg",
                f"{prefix}_tail_confidence",
                f"{prefix}_tail_avg_over_prior_avg",
                f"{prefix}_tail_avg_minus_low_avg",
                f"{prefix}_tail_obs_log",
            ]
        )
    return names

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

DERIVED_NUMERIC_FEATURES = [
    "display_capacity",
    "merchandised_display_capacity",
    "log_store_prior_obs",
    "log_store_subsegment_prior_obs",
    "log_chain_segment_prior_obs",
    "log_segment_prior_obs",
    "store_prior_avg_over_global_avg",
    "store_subsegment_avg_over_store_avg",
    "store_subsegment_avg_over_segment_avg",
    "chain_segment_avg_over_segment_avg",
    "chain_segment_avg_over_global_avg",
    "segment_avg_over_global_avg",
    "store_prior_rate_over_global_rate",
    "store_subsegment_rate_over_store_rate",
    "chain_segment_rate_over_segment_rate",
    "display_x_store_prior_avg",
    "display_x_chain_segment_avg",
    "merchandised_x_chain_segment_avg",
    "pockets_x_store_prior_avg",
    "facings_x_chain_segment_avg",
    "store_chain_subsegment_tail_signal",
    "embedding_p90_over_store_subsegment_avg",
    "embedding_p90_over_chain_segment_avg",
    "embedding_p90_x_store_subsegment_avg",
    "embedding_p90_x_chain_segment_avg",
    "embedding_tail_confidence",
    "embedding_tail_uncertainty_confidence",
    "embedding_tail_context_signal",
    "embedding_tail_context_confidence",
    *embedding_affinity_derived_feature_names(),
]

BASE_PRIOR_SPECS: dict[str, list[str]] = {
    "global": [],
    "chain": ["store_chain"],
    "class": ["classoftrade"],
    "store": ["store_id"],
    "store_title": ["store_id", "title"],
    "store_segment": ["store_id", "segment"],
    "store_subsegment": ["store_id", "subsegment"],
    "store_type": ["store_id", "type"],
    "chain_segment": ["store_chain", "segment"],
    "chain_class_subsegment": ["store_chain", "classoftrade", "subsegment"],
    "chain_title": ["store_chain", "title"],
    "class_subsegment": ["classoftrade", "subsegment"],
    "class_title": ["classoftrade", "title"],
    "title": ["title"],
    "segment": ["segment"],
    "subsegment": ["subsegment"],
}
WINDOW_PRIOR_DAYS = (90, 180, 365)
WINDOW_PRIOR_SPECS: dict[str, list[str]] = {
    "global": [],
    "store": ["store_id"],
    "store_segment": ["store_id", "segment"],
    "store_subsegment": ["store_id", "subsegment"],
    "chain_segment": ["store_chain", "segment"],
    "chain_class_subsegment": ["store_chain", "classoftrade", "subsegment"],
    "class_subsegment": ["classoftrade", "subsegment"],
}
RECENCY_PRIOR_HALF_LIFE_DAYS = 180
RECENCY_PRIOR_SPECS = WINDOW_PRIOR_SPECS
MONTH_PRIOR_SPECS: dict[str, list[str]] = {
    "global_month": ["onsale_month"],
    "segment_month": ["segment", "onsale_month"],
    "subsegment_month": ["subsegment", "onsale_month"],
    "store_subsegment_month": ["store_id", "subsegment", "onsale_month"],
    "chain_segment_month": ["store_chain", "segment", "onsale_month"],
}


def prior_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_prior_positive_rate",
        f"{prefix}_prior_positive_avg_sales",
        f"{prefix}_prior_obs",
    ]


def window_prior_feature_names(prefix: str, days: int) -> list[str]:
    return [
        f"{prefix}_completed_{days}d_positive_rate",
        f"{prefix}_completed_{days}d_positive_avg_sales",
        f"{prefix}_completed_{days}d_obs",
    ]


def recency_prior_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_completed_recency_hl{RECENCY_PRIOR_HALF_LIFE_DAYS}_positive_rate",
        f"{prefix}_completed_recency_hl{RECENCY_PRIOR_HALF_LIFE_DAYS}_positive_avg_sales",
        f"{prefix}_completed_recency_hl{RECENCY_PRIOR_HALF_LIFE_DAYS}_obs",
    ]


def completed_prior_feature_names() -> list[str]:
    names: list[str] = []
    for prefix in BASE_PRIOR_SPECS:
        names.extend(prior_feature_names(prefix))
    for prefix in WINDOW_PRIOR_SPECS:
        for days in WINDOW_PRIOR_DAYS:
            names.extend(window_prior_feature_names(prefix, days))
    for prefix in RECENCY_PRIOR_SPECS:
        names.extend(recency_prior_feature_names(prefix))
    for prefix in MONTH_PRIOR_SPECS:
        names.extend(prior_feature_names(prefix))
    return names


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
    *completed_prior_feature_names(),
    *EMBEDDING_ANALOG_FEATURES,
    *embedding_affinity_feature_names(),
    *DERIVED_NUMERIC_FEATURES,
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


def safe_series_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_derived_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    display_capacity = frame["facings"].astype(float) * frame["pockets"].astype(float)
    frame["display_capacity"] = display_capacity
    frame["merchandised_display_capacity"] = display_capacity * frame["merchandised"].astype(float)
    frame["log_store_prior_obs"] = np.log1p(frame["store_prior_obs"].astype(float))
    frame["log_store_subsegment_prior_obs"] = np.log1p(frame["store_subsegment_prior_obs"].astype(float))
    frame["log_chain_segment_prior_obs"] = np.log1p(frame["chain_segment_prior_obs"].astype(float))
    frame["log_segment_prior_obs"] = np.log1p(frame["segment_prior_obs"].astype(float))

    frame["store_prior_avg_over_global_avg"] = safe_series_ratio(
        frame["store_prior_positive_avg_sales"],
        frame["global_prior_positive_avg_sales"],
    )
    frame["store_subsegment_avg_over_store_avg"] = safe_series_ratio(
        frame["store_subsegment_prior_positive_avg_sales"],
        frame["store_prior_positive_avg_sales"],
    )
    frame["store_subsegment_avg_over_segment_avg"] = safe_series_ratio(
        frame["store_subsegment_prior_positive_avg_sales"],
        frame["segment_prior_positive_avg_sales"],
    )
    frame["chain_segment_avg_over_segment_avg"] = safe_series_ratio(
        frame["chain_segment_prior_positive_avg_sales"],
        frame["segment_prior_positive_avg_sales"],
    )
    frame["chain_segment_avg_over_global_avg"] = safe_series_ratio(
        frame["chain_segment_prior_positive_avg_sales"],
        frame["global_prior_positive_avg_sales"],
    )
    frame["segment_avg_over_global_avg"] = safe_series_ratio(
        frame["segment_prior_positive_avg_sales"],
        frame["global_prior_positive_avg_sales"],
    )
    frame["store_prior_rate_over_global_rate"] = safe_series_ratio(
        frame["store_prior_positive_rate"],
        frame["global_prior_positive_rate"],
    )
    frame["store_subsegment_rate_over_store_rate"] = safe_series_ratio(
        frame["store_subsegment_prior_positive_rate"],
        frame["store_prior_positive_rate"],
    )
    frame["chain_segment_rate_over_segment_rate"] = safe_series_ratio(
        frame["chain_segment_prior_positive_rate"],
        frame["segment_prior_positive_rate"],
    )

    frame["display_x_store_prior_avg"] = display_capacity * frame["store_prior_positive_avg_sales"].astype(float)
    frame["display_x_chain_segment_avg"] = display_capacity * frame["chain_segment_prior_positive_avg_sales"].astype(float)
    frame["merchandised_x_chain_segment_avg"] = (
        frame["merchandised"].astype(float) * frame["chain_segment_prior_positive_avg_sales"].astype(float)
    )
    frame["pockets_x_store_prior_avg"] = frame["pockets"].astype(float) * frame["store_prior_positive_avg_sales"].astype(float)
    frame["facings_x_chain_segment_avg"] = frame["facings"].astype(float) * frame["chain_segment_prior_positive_avg_sales"].astype(float)
    frame["store_chain_subsegment_tail_signal"] = (
        frame["store_prior_avg_over_global_avg"]
        * frame["chain_segment_avg_over_global_avg"]
        * np.log1p(frame["store_subsegment_prior_obs"].astype(float))
    )

    analog_p90 = frame[f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p90_positive_sales"].astype(float)
    analog_uncertainty = frame[f"embedding_analog_k{EMBEDDING_ANALOG_K}_shrunk_p90_minus_p50_positive_sales"].astype(float)
    analog_effective_neighbors = frame[f"embedding_analog_k{EMBEDDING_ANALOG_K}_effective_neighbor_count"].astype(float)
    frame["embedding_p90_over_store_subsegment_avg"] = safe_series_ratio(
        analog_p90,
        frame["store_subsegment_prior_positive_avg_sales"],
    )
    frame["embedding_p90_over_chain_segment_avg"] = safe_series_ratio(
        analog_p90,
        frame["chain_segment_prior_positive_avg_sales"],
    )
    frame["embedding_p90_x_store_subsegment_avg"] = analog_p90 * frame["store_subsegment_prior_positive_avg_sales"].astype(float)
    frame["embedding_p90_x_chain_segment_avg"] = analog_p90 * frame["chain_segment_prior_positive_avg_sales"].astype(float)
    frame["embedding_tail_confidence"] = analog_p90 * np.log1p(analog_effective_neighbors)
    frame["embedding_tail_uncertainty_confidence"] = analog_uncertainty * np.log1p(analog_effective_neighbors)
    context_tail = frame["chain_class_subsegment_affinity_tail_minus_low_similarity"].astype(float)
    context_tail_obs = frame["chain_class_subsegment_affinity_tail_obs"].astype(float)
    frame["embedding_tail_context_signal"] = analog_p90 * context_tail
    frame["embedding_tail_context_confidence"] = context_tail * np.log1p(context_tail_obs)
    affinity_derived_columns: dict[str, pd.Series] = {}
    for prefix, (prior_avg_col, prior_obs_col) in EMBEDDING_AFFINITY_PRIOR_CONTEXTS.items():
        tail_signal = frame[f"{prefix}_tail_minus_low_similarity"].astype(float)
        tail_obs = frame[f"{prefix}_tail_obs"].astype(float)
        tail_avg = frame[f"{prefix}_tail_avg_sales"].astype(float)
        low_avg = frame[f"{prefix}_low_avg_sales"].astype(float)
        prior_avg = frame[prior_avg_col].astype(float)
        prior_obs = frame[prior_obs_col].astype(float)
        confidence = np.log1p(tail_obs)
        affinity_derived_columns[f"{prefix}_tail_signal_x_analog_p90"] = tail_signal * analog_p90 * confidence
        affinity_derived_columns[f"{prefix}_tail_signal_x_prior_avg"] = tail_signal * prior_avg * np.log1p(prior_obs)
        affinity_derived_columns[f"{prefix}_tail_confidence"] = frame[f"{prefix}_tail_similarity"].astype(float) * confidence
        affinity_derived_columns[f"{prefix}_tail_avg_over_prior_avg"] = safe_series_ratio(tail_avg, prior_avg)
        affinity_derived_columns[f"{prefix}_tail_avg_minus_low_avg"] = tail_avg - low_avg
        affinity_derived_columns[f"{prefix}_tail_obs_log"] = confidence
    if affinity_derived_columns:
        frame = pd.concat([frame, pd.DataFrame(affinity_derived_columns, index=frame.index)], axis=1)

    frame[DERIVED_NUMERIC_FEATURES] = frame[DERIVED_NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return frame


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


def asof_context_merge(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    left = frame[[*group_cols, "onsaledate"]].copy()
    left["_row_id"] = np.arange(len(frame), dtype=np.int64)
    if history.empty:
        return pd.DataFrame(0.0, index=np.arange(len(frame)), columns=value_cols, dtype=np.float32)
    right = history[[*group_cols, "completion_date", *value_cols]].copy()
    for col in group_cols:
        left[col] = left[col].astype("string").fillna("UNKNOWN").astype(str)
        right[col] = right[col].astype("string").fillna("UNKNOWN").astype(str)
    left = left.sort_values(["onsaledate", *group_cols] if group_cols else ["onsaledate"], kind="mergesort")
    right = right.sort_values(["completion_date", *group_cols] if group_cols else ["completion_date"], kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on="onsaledate",
        right_on="completion_date",
        by=group_cols if group_cols else None,
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_row_id", kind="mergesort")
    return merged[value_cols].reset_index(drop=True).fillna(0.0)


def cumulative_embedding_context(
    frame: pd.DataFrame,
    row_vectors: np.ndarray,
    group_cols: list[str],
    mask: np.ndarray,
    weight: np.ndarray,
) -> pd.DataFrame:
    vector_cols = [f"_affinity_v{idx:02d}" for idx in range(row_vectors.shape[1])]
    source = frame.loc[mask, [*group_cols, "offsaledate", "sales_target"]].copy()
    result_cols = [*group_cols, "completion_date", *vector_cols, "_obs", "_sales_sum", "_weight_sum"]
    if source.empty:
        return pd.DataFrame(columns=result_cols)
    source["completion_date"] = pd.to_datetime(source["offsaledate"])
    source["_obs"] = 1.0
    source["_sales_sum"] = source["sales_target"].astype(np.float32)
    source["_weight_sum"] = weight[mask].astype(np.float32)
    weighted_vectors = row_vectors[mask] * source["_weight_sum"].to_numpy(dtype=np.float32)[:, None]
    for idx, col in enumerate(vector_cols):
        source[col] = weighted_vectors[:, idx]
    group_keys = [*group_cols, "completion_date"]
    aggregate = (
        source.groupby(group_keys, dropna=False, observed=True)
        .agg(
            **{col: (col, "sum") for col in vector_cols},
            _obs=("_obs", "sum"),
            _sales_sum=("_sales_sum", "sum"),
            _weight_sum=("_weight_sum", "sum"),
        )
        .reset_index()
        .sort_values(group_keys, kind="mergesort")
    )
    cumulative_cols = [*vector_cols, "_obs", "_sales_sum", "_weight_sum"]
    if group_cols:
        aggregate[cumulative_cols] = aggregate.groupby(group_cols, dropna=False, observed=True)[cumulative_cols].cumsum()
    else:
        aggregate[cumulative_cols] = aggregate[cumulative_cols].cumsum()
    return aggregate


def context_similarity_features(
    current_vectors: np.ndarray,
    merged: pd.DataFrame,
    stem: str,
) -> pd.DataFrame:
    vector_cols = [f"_affinity_v{idx:02d}" for idx in range(current_vectors.shape[1])]
    sums = merged[vector_cols].to_numpy(dtype=np.float32)
    weight_sum = merged["_weight_sum"].to_numpy(dtype=np.float32)
    obs = merged["_obs"].to_numpy(dtype=np.float32)
    sales_sum = merged["_sales_sum"].to_numpy(dtype=np.float32)
    centroid_norm = np.linalg.norm(sums, axis=1)
    similarity = np.zeros(len(merged), dtype=np.float32)
    valid = (weight_sum > 0) & (centroid_norm > 0)
    if valid.any():
        similarity[valid] = (current_vectors[valid] * sums[valid]).sum(axis=1) / centroid_norm[valid]
    avg_sales = np.divide(sales_sum, obs, out=np.zeros_like(sales_sum, dtype=np.float32), where=obs > 0)
    return pd.DataFrame(
        {
            f"{stem}_similarity": similarity,
            f"{stem}_obs": obs,
            f"{stem}_avg_sales": avg_sales,
        }
    )


def add_embedding_affinity_features(
    frame: pd.DataFrame,
    row_reduced: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = frame.copy()
    train_targets = frame.loc[frame["split"] == "train", "sales_target"].astype(float)
    low_threshold = float(train_targets.quantile(EMBEDDING_AFFINITY_LOW_QUANTILE))
    tail_threshold = float(train_targets.quantile(EMBEDDING_AFFINITY_TAIL_QUANTILE))
    current_vectors = row_normalize(row_reduced[:, :EMBEDDING_AFFINITY_COMPONENTS].astype(np.float32))
    low_mask = frame["sales_target"].to_numpy(dtype=np.float32) <= low_threshold
    tail_mask = frame["sales_target"].to_numpy(dtype=np.float32) >= tail_threshold
    low_weight = np.ones(len(frame), dtype=np.float32)
    tail_weight = frame["sales_target"].to_numpy(dtype=np.float32)
    summary: dict[str, object] = {
        "method": "completed_context_tail_low_embedding_centroid_affinity",
        "components": EMBEDDING_AFFINITY_COMPONENTS,
        "low_quantile": EMBEDDING_AFFINITY_LOW_QUANTILE,
        "tail_quantile": EMBEDDING_AFFINITY_TAIL_QUANTILE,
        "low_sales_threshold": low_threshold,
        "tail_sales_threshold": tail_threshold,
        "contexts": {},
        "features": embedding_affinity_feature_names(),
        "completion_rule": "historical product offsaledate must be strictly before current row onsaledate",
    }

    affinity_columns: dict[str, np.ndarray] = {}
    for prefix, group_cols in EMBEDDING_AFFINITY_CONTEXTS.items():
        log(f"Building embedding affinity features for {prefix}")
        tail_history = cumulative_embedding_context(frame, current_vectors, group_cols, tail_mask, tail_weight)
        low_history = cumulative_embedding_context(frame, current_vectors, group_cols, low_mask, low_weight)
        tail_cols = [col for col in tail_history.columns if col.startswith("_affinity_v") or col in {"_obs", "_sales_sum", "_weight_sum"}]
        low_cols = [col for col in low_history.columns if col.startswith("_affinity_v") or col in {"_obs", "_sales_sum", "_weight_sum"}]
        tail_merged = asof_context_merge(frame, tail_history, group_cols, tail_cols)
        low_merged = asof_context_merge(frame, low_history, group_cols, low_cols)
        tail_features = context_similarity_features(current_vectors, tail_merged, f"{prefix}_tail")
        low_features = context_similarity_features(current_vectors, low_merged, f"{prefix}_low")
        tail_similarity = tail_features[f"{prefix}_tail_similarity"].to_numpy(dtype=np.float32)
        low_similarity = low_features[f"{prefix}_low_similarity"].to_numpy(dtype=np.float32)
        tail_obs = tail_features[f"{prefix}_tail_obs"].to_numpy(dtype=np.float32)
        low_obs = low_features[f"{prefix}_low_obs"].to_numpy(dtype=np.float32)
        affinity_columns[f"{prefix}_tail_similarity"] = tail_similarity
        affinity_columns[f"{prefix}_low_similarity"] = low_similarity
        affinity_columns[f"{prefix}_tail_minus_low_similarity"] = (tail_similarity - low_similarity).astype(np.float32)
        affinity_columns[f"{prefix}_tail_obs"] = tail_obs
        affinity_columns[f"{prefix}_low_obs"] = low_obs
        affinity_columns[f"{prefix}_tail_avg_sales"] = tail_features[f"{prefix}_tail_avg_sales"].to_numpy(dtype=np.float32)
        affinity_columns[f"{prefix}_low_avg_sales"] = low_features[f"{prefix}_low_avg_sales"].to_numpy(dtype=np.float32)
        summary["contexts"][prefix] = {
            "group_columns": group_cols,
            "tail_history_rows": int(len(tail_history)),
            "low_history_rows": int(len(low_history)),
            "rows_with_tail_context": int((tail_obs > 0).sum()),
            "rows_with_low_context": int((low_obs > 0).sum()),
        }

    if affinity_columns:
        frame = pd.concat([frame, pd.DataFrame(affinity_columns, index=frame.index)], axis=1)
    features = embedding_affinity_feature_names()
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return frame, summary


def add_embedding_features(
    frame: pd.DataFrame,
    family: str,
    db_uri: str,
    family_dir: Path,
    stage: str = "regressor_positive",
) -> tuple[pd.DataFrame, dict[str, object]]:
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
    frame, affinity_summary = add_embedding_affinity_features(frame, row_reduced)

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
        "affinity_features": affinity_summary,
    }
    artifact_path = family_dir / f"embedding_pca_{stage}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return frame, {
        key: value
        for key, value in artifact.items()
        if key not in {"mean", "components", "explained_variance_ratio", "cumulative_explained_variance"}
    } | {
        "artifact_path": str(artifact_path),
        "cumulative_explained_variance_at_130": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "analog_features": analog_summary,
        "affinity_features": affinity_summary,
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
target as (
    select *
    from base_all
    where positive_sale_flag = 1
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
from target t
order by t.onsaledate, t.product_id, t.store_id
"""


HISTORY_GROUP_SQL = {
    "store_id": "fs.store_id",
    "title": category_sql("dp.title"),
    "type": category_sql("dp.type"),
    "segment": category_sql("dp.segment"),
    "subsegment": category_sql("dp.subsegment"),
    "store_chain": category_sql("ds.store_chain"),
    "classoftrade": category_sql("ds.classoftrade"),
    "onsale_month": "extract(month from dp.onsaledate)::int",
}


def history_aggregate_query(family: str, group_cols: list[str]) -> str:
    select_cols = []
    group_exprs = []
    for col in group_cols:
        expr = HISTORY_GROUP_SQL[col]
        select_cols.append(f"{expr} as {col}")
        group_exprs.append(expr)
    prefix = ",\n        ".join(select_cols)
    if prefix:
        prefix = f"{prefix},\n        "
    group_by = ", ".join([*group_exprs, "dp.offsaledate"])
    if not group_by:
        group_by = "dp.offsaledate"
    return f"""
    select
        {prefix}dp.offsaledate::date as completion_date,
        coalesce(sum(greatest(fs.soldqty, 0)) filter (where fs.soldqty > 0), 0)::double precision as positive_sales_sum,
        coalesce(sum(case when fs.soldqty > 0 then 1 else 0 end), 0)::double precision as positive_row_count,
        count(*)::double precision as row_count
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    join core.dim_store ds on ds.store_id = fs.store_id
    where dp.type = '{family}'
      and fs.drawqty <> 0
    group by {group_by}
    """


def coerce_group_columns(frame: pd.DataFrame, history: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    history = history.copy()
    for col in group_cols:
        if col in {"store_id", "onsale_month"}:
            frame[col] = frame[col].astype(np.int64)
            history[col] = history[col].astype(np.int64)
        else:
            frame[col] = frame[col].astype("string").fillna("UNKNOWN").astype(str)
            history[col] = history[col].astype("string").fillna("UNKNOWN").astype(str)
    return frame, history


def cumulative_history(history: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    history = history.sort_values([*group_cols, "completion_date"] if group_cols else ["completion_date"]).copy()
    if group_cols:
        grouped = history.groupby(group_cols, observed=True, sort=False)
        for col in value_cols:
            history[f"cum_{col}"] = grouped[col].cumsum()
    else:
        for col in value_cols:
            history[f"cum_{col}"] = history[col].cumsum()
    return history


def merge_asof_history(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    group_cols: list[str],
    left_date_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    left_cols = [*group_cols, left_date_col]
    left = frame[left_cols].copy()
    left["_row_id"] = np.arange(len(frame), dtype=np.int64)
    right = history[[*group_cols, "completion_date", *value_cols]].copy()

    left_sort_cols = [left_date_col, *group_cols]
    right_sort_cols = ["completion_date", *group_cols]
    left = left.sort_values(left_sort_cols, kind="mergesort")
    right = right.sort_values(right_sort_cols, kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on=left_date_col,
        right_on="completion_date",
        by=group_cols if group_cols else None,
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_row_id", kind="mergesort")
    return merged[value_cols].reset_index(drop=True).fillna(0.0)


def assign_prior_metrics(frame: pd.DataFrame, prefix: str, sums: pd.DataFrame, feature_names: list[str]) -> None:
    positive_sales = sums["positive_sales_sum"].to_numpy(dtype=np.float64)
    positive_rows = sums["positive_row_count"].to_numpy(dtype=np.float64)
    rows = sums["row_count"].to_numpy(dtype=np.float64)
    frame[feature_names[0]] = safe_series_ratio(pd.Series(positive_rows), pd.Series(rows)).to_numpy(dtype=np.float32)
    frame[feature_names[1]] = safe_series_ratio(pd.Series(positive_sales), pd.Series(positive_rows)).to_numpy(dtype=np.float32)
    frame[feature_names[2]] = rows.astype(np.float32)


def add_base_completed_prior(frame: pd.DataFrame, history: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    value_cols = ["positive_sales_sum", "positive_row_count", "row_count"]
    history = cumulative_history(history, group_cols, value_cols)
    cumulative_cols = [f"cum_{col}" for col in value_cols]
    merged = merge_asof_history(frame, history, group_cols, "onsaledate", cumulative_cols)
    merged.columns = value_cols
    assign_prior_metrics(frame, prefix, merged, prior_feature_names(prefix))
    return history


def add_window_completed_prior(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    days: int,
) -> None:
    value_cols = ["positive_sales_sum", "positive_row_count", "row_count"]
    cumulative_cols = [f"cum_{col}" for col in value_cols]
    current = merge_asof_history(frame, history, group_cols, "onsaledate", cumulative_cols)
    frame["_prior_window_start"] = frame["onsaledate"] - pd.to_timedelta(days, unit="D")
    start = merge_asof_history(frame, history, group_cols, "_prior_window_start", cumulative_cols)
    window = (current - start).clip(lower=0.0)
    window.columns = value_cols
    assign_prior_metrics(frame, prefix, window, window_prior_feature_names(prefix, days))
    frame.drop(columns=["_prior_window_start"], inplace=True)


def add_recency_completed_prior(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    origin_date: pd.Timestamp,
) -> None:
    value_cols = ["weighted_positive_sales_sum", "weighted_positive_row_count", "weighted_row_count"]
    half_life = float(RECENCY_PRIOR_HALF_LIFE_DAYS)
    history = history.copy()
    completion_days = (history["completion_date"] - origin_date).dt.days.astype(np.float64)
    growth = np.exp(completion_days / half_life)
    history["weighted_positive_sales_sum"] = history["positive_sales_sum"].astype(float) * growth
    history["weighted_positive_row_count"] = history["positive_row_count"].astype(float) * growth
    history["weighted_row_count"] = history["row_count"].astype(float) * growth
    history = cumulative_history(history, group_cols, value_cols)
    cumulative_cols = [f"cum_{col}" for col in value_cols]
    merged = merge_asof_history(frame, history, group_cols, "onsaledate", cumulative_cols)
    current_days = (frame["onsaledate"] - origin_date).dt.days.astype(np.float64).to_numpy()
    decay = np.exp(-current_days / half_life)
    sums = pd.DataFrame(
        {
            "positive_sales_sum": merged[cumulative_cols[0]].to_numpy(dtype=np.float64) * decay,
            "positive_row_count": merged[cumulative_cols[1]].to_numpy(dtype=np.float64) * decay,
            "row_count": merged[cumulative_cols[2]].to_numpy(dtype=np.float64) * decay,
        }
    )
    assign_prior_metrics(frame, prefix, sums, recency_prior_feature_names(prefix))


def add_completed_prior_features(frame: pd.DataFrame, family: str, db_uri: str) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = frame.copy()
    frame["onsaledate"] = pd.to_datetime(frame["onsaledate"])
    specs: dict[str, list[str]] = {}
    specs.update(BASE_PRIOR_SPECS)
    specs.update(WINDOW_PRIOR_SPECS)
    specs.update(RECENCY_PRIOR_SPECS)
    specs.update(MONTH_PRIOR_SPECS)
    prior_summary = {
        "method": "completed_issue_history",
        "completion_rule": "historical product offsaledate must be strictly before current product onsaledate",
        "base_prior_specs": BASE_PRIOR_SPECS,
        "window_prior_days": WINDOW_PRIOR_DAYS,
        "window_prior_specs": WINDOW_PRIOR_SPECS,
        "recency_prior_half_life_days": RECENCY_PRIOR_HALF_LIFE_DAYS,
        "recency_prior_specs": RECENCY_PRIOR_SPECS,
        "month_prior_specs": MONTH_PRIOR_SPECS,
        "features": completed_prior_feature_names(),
        "history_frames": {},
    }
    origin_date: pd.Timestamp | None = None

    for prefix, group_cols in specs.items():
        log(f"Loading {family} completed-history aggregates for {prefix}")
        history = run_psql_csv(db_uri, history_aggregate_query(family, group_cols))
        history["completion_date"] = pd.to_datetime(history["completion_date"])
        frame, history = coerce_group_columns(frame, history, group_cols)
        if origin_date is None:
            origin_date = pd.Timestamp(history["completion_date"].min())
        prior_summary["history_frames"][prefix] = {
            "group_columns": group_cols,
            "rows": int(len(history)),
            "min_completion_date": pd.Timestamp(history["completion_date"].min()).strftime("%Y-%m-%d"),
            "max_completion_date": pd.Timestamp(history["completion_date"].max()).strftime("%Y-%m-%d"),
        }

        cumulative = None
        if prefix in BASE_PRIOR_SPECS:
            cumulative = add_base_completed_prior(frame, history, group_cols, prefix)
        if prefix in WINDOW_PRIOR_SPECS:
            if cumulative is None:
                cumulative = cumulative_history(history, group_cols, ["positive_sales_sum", "positive_row_count", "row_count"])
            for days in WINDOW_PRIOR_DAYS:
                add_window_completed_prior(frame, cumulative, group_cols, prefix, days)
        if prefix in RECENCY_PRIOR_SPECS:
            assert origin_date is not None
            add_recency_completed_prior(frame, history, group_cols, prefix, origin_date)
        if prefix in MONTH_PRIOR_SPECS:
            add_base_completed_prior(frame, history, group_cols, prefix)

    feature_names = completed_prior_feature_names()
    frame[feature_names] = frame[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return frame, prior_summary


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
    frame, prior_summary = add_completed_prior_features(frame, family, db_uri)
    frame, embedding_summary = add_embedding_features(frame, family, db_uri, family_dir)
    frame = add_derived_numeric_features(frame)
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
        "completed_prior_features": prior_summary,
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
