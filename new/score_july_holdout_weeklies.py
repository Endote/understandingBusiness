#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from build_modeling_datasets import (
    BASE_DEMOGRAPHIC_COLUMNS,
    DEFAULT_DB_URI,
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
    EMBEDDING_AFFINITY_COMPONENTS,
    EMBEDDING_AFFINITY_CONTEXTS,
    EMBEDDING_AFFINITY_LOW_QUANTILE,
    EMBEDDING_AFFINITY_TAIL_QUANTILE,
    EMBEDDING_ANALOG_FEATURES,
    EMBEDDING_ANALOG_K,
    EMBEDDING_FEATURES,
    add_completed_prior_features,
    add_derived_numeric_features,
    aggregate_prior_stats,
    asof_context_merge,
    build_regressor_positive_query,
    category_sql,
    context_similarity_features,
    cumulative_embedding_context,
    effective_neighbor_count,
    load_family_embeddings,
    parse_embedding,
    row_normalize,
    run_psql_copy,
    run_psql_csv,
    safe_ratio,
    shrink_stats,
)
from calibrate_incidence_classifier import (
    apply_group_multiplier,
    assign_probability_bin,
    artifact_from_payload as incidence_artifact_from_payload,
    probability_bins_from_validation,
    predict_probability,
)
from evaluate_incidence_pipeline import (
    amount_rank_bucket_multiplier,
    apply_tail_multiplier,
    load_optional_band_model,
    load_optional_low_model,
    load_tail_model,
    tail_multiplier_from_summary,
)
from train_stage_model import EncodingArtifact, predict_frame, transform_frame
from train_tail_layer import load_base_regressor, probability_rank


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "JulyScoring"
DEFAULT_AMOUNT_RUN_DIR = NEW_DIR / "output" / "model_runs" / "weeklies" / "regressor_positive" / "asym_curve_log1p" / "20260517_205934"
DEFAULT_TAIL_RUN_DIR = NEW_DIR / "output" / "tail_layer_runs" / "weeklies" / "regressor_positive" / "tail_q70" / "20260517_210107"
DEFAULT_CLASSIFIER_RUN_DIR = NEW_DIR / "output" / "model_runs" / "weeklies" / "classifier_all" / "binary_logistic" / "20260517_211136"
DEFAULT_CALIBRATION_RUN_DIR = NEW_DIR / "output" / "model_runs" / "weeklies" / "classifier_all" / "calibration" / "20260517_213616"
DEFAULT_PIPELINE_VARIANT = "G_soft_gate_amount_calibrated_t0.70_floor0.00_low0.85_mid1.10_top0.95_scale0.95"
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]
META_COLUMNS = [
    "store_id",
    "product_id",
    "onsaledate",
    "offsaledate",
    "title",
    "type",
    "segment",
    "subsegment",
    "frequency",
    "store_chain",
    "region",
    "classoftrade",
    "price",
    "issue_length_days",
    "onsale_month",
    "onsale_week",
    "onsale_dow",
    "merchandised",
    "facings",
    "pockets",
]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_holdout_query(family: str) -> str:
    demo_cols = ",\n        ".join(f"dg.{col}" for col in BASE_DEMOGRAPHIC_COLUMNS)
    demo_select = ",\n    ".join(f"coalesce(t.{col}, 0) as {col}" for col in BASE_DEMOGRAPHIC_COLUMNS)
    return f"""
with base_all as (
    select
        ps.store_id,
        ps.product_id,
        0::double precision as sales_target,
        0::double precision as soldqty_raw,
        0::double precision as drawqty,
        0::int as positive_sale_flag,
        0::int as negative_sales_flag,
        0::int as stockout_proxy_flag,
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
    from holdout.printing_schedule ps
    join holdout.dim_product dp on dp.product_id = ps.product_id
    join holdout.dim_store ds on ds.store_id = ps.store_id
    left join holdout.demographic dg on dg.postal_code = ds.postal_code
    where dp.type = '{family}'
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


def load_holdout_frame(db_uri: str, family: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="paperrush_july_holdout_") as tmp_dir:
        csv_path = Path(tmp_dir) / f"{family.lower()}_july_holdout.csv"
        log(f"Exporting {family} July holdout rows from Postgres")
        run_psql_copy(db_uri, build_holdout_query(family), csv_path)
        frame = pd.read_csv(csv_path, parse_dates=["onsaledate", "offsaledate"])
    frame["split"] = "score"
    return frame


def holdout_embedding_query(family: str) -> str:
    return f"""
    select
        dp.product_id,
        dp.onsaledate,
        coalesce(dp.offsaledate, dp.onsaledate) as offsaledate,
        {category_sql("dp.subsegment")} as subsegment,
        ce.embedding::text as embedding
    from holdout.dim_product dp
    left join holdout.content_embedding ce on ce.product_id = dp.product_id
    where dp.type = '{family}'
    order by dp.onsaledate, dp.product_id
    """


def load_holdout_embeddings(db_uri: str, family: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    frame = run_psql_csv(db_uri, holdout_embedding_query(family))
    frame["onsaledate"] = pd.to_datetime(frame["onsaledate"])
    frame["offsaledate"] = pd.to_datetime(frame["offsaledate"])
    missing = frame.loc[frame["embedding"].isna(), ["product_id", "onsaledate", "subsegment"]].copy()
    matrix = np.zeros((len(frame), 384), dtype=np.float64)
    valid = frame["embedding"].notna().to_numpy()
    if valid.any():
        matrix[valid] = np.vstack(frame.loc[valid, "embedding"].map(parse_embedding).to_numpy())
    frame = frame.drop(columns=["embedding"])
    frame["has_embedding"] = valid
    return frame, matrix, missing


def pca_transform_from_artifact(matrix: np.ndarray, pca_artifact: dict[str, object]) -> np.ndarray:
    mean = np.asarray(pca_artifact["mean"], dtype=np.float64)
    components = np.asarray(pca_artifact["components"], dtype=np.float64)
    return ((matrix - mean) @ components.T).astype(np.float32)


def build_holdout_embedding_analog_features(
    core_products: pd.DataFrame,
    core_matrix: np.ndarray,
    holdout_products: pd.DataFrame,
    holdout_matrix: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    core_products = core_products.reset_index(drop=True)
    holdout_products = holdout_products.reset_index(drop=True)
    core_dates = core_products["onsaledate"].to_numpy(dtype="datetime64[ns]")
    core_offsale_dates = core_products["offsaledate"].to_numpy(dtype="datetime64[ns]")
    core_subsegments = core_products["subsegment"].astype(str).to_numpy()
    observed_rows = core_products["observed_rows"].to_numpy(dtype=np.float64)
    positive_rows = core_products["positive_rows"].to_numpy(dtype=np.float64)
    positive_sales_sum = core_products["positive_sales_sum"].to_numpy(dtype=np.float64)
    core_normalized = row_normalize(core_matrix)
    holdout_normalized = row_normalize(holdout_matrix)
    stats = core_products[
        [
            "avg_positive_sales",
            "positive_sale_rate",
            "p50_positive_sales",
            "p75_positive_sales",
            "p90_positive_sales",
        ]
    ].to_numpy(dtype=np.float64)
    quantile_stats = core_products[
        [
            "p50_positive_sales",
            "p75_positive_sales",
            "p90_positive_sales",
        ]
    ].to_numpy(dtype=np.float64)
    result = np.zeros((len(holdout_products), len(EMBEDDING_ANALOG_FEATURES)), dtype=np.float32)

    for idx, row in holdout_products.iterrows():
        if not bool(row["has_embedding"]):
            continue
        current_date = np.datetime64(row["onsaledate"], "ns")
        candidate_indices = np.flatnonzero((core_dates < current_date) & (core_offsale_dates < current_date))
        if len(candidate_indices) == 0:
            continue
        similarities = core_normalized[candidate_indices] @ holdout_normalized[idx]
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
        weighted_stats = np.average(stats[neighbor_indices], axis=0, weights=weights)
        effective_count = effective_neighbor_count(weights)
        same_subsegment = candidate_indices[core_subsegments[candidate_indices] == str(row["subsegment"])]
        prior_indices = same_subsegment if len(same_subsegment) else candidate_indices
        subsegment_prior_stats = aggregate_prior_stats(
            prior_indices,
            positive_sales_sum=positive_sales_sum,
            positive_rows=positive_rows,
            observed_rows=observed_rows,
            quantile_stats=quantile_stats,
        )
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
        result[idx] = np.array(
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
                float(weighted_stats[4] - weighted_stats[2]),
                safe_ratio(float(weighted_stats[4]), float(weighted_stats[0])),
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
                float(shrunk[4] - shrunk[2]),
                safe_ratio(float(shrunk[4]), float(shrunk[1])),
            ],
            dtype=np.float32,
        )

    analog_frame = pd.DataFrame({"product_id": holdout_products["product_id"].to_numpy(dtype=np.int64)})
    for feature_idx, feature in enumerate(EMBEDDING_ANALOG_FEATURES):
        analog_frame[feature] = result[:, feature_idx]
    summary = {
        "method": "holdout_products_against_completed_core_same_family_prior_product_cosine_neighbors",
        "neighbor_k": EMBEDDING_ANALOG_K,
        "holdout_products": int(len(holdout_products)),
        "holdout_products_with_embedding": int(holdout_products["has_embedding"].sum()),
        "holdout_products_without_embedding": int((~holdout_products["has_embedding"]).sum()),
        "products_with_prior_neighbors": int((result[:, 0] > 0).sum()),
        "products_without_prior_neighbors": int((result[:, 0] == 0).sum()),
    }
    return analog_frame, summary


def assign_split_from_manifest(frame: pd.DataFrame, manifest: dict[str, object]) -> pd.DataFrame:
    split_dates = manifest["split_dates"]
    mapping = {}
    for split, key in (("train", "train_dates"), ("valid", "validation_dates"), ("test", "test_dates")):
        mapping.update({date: split for date in split_dates[key]})
    result = frame.copy()
    result["split"] = result["onsaledate"].dt.strftime("%Y-%m-%d").map(mapping).fillna("history")
    return result


def load_positive_history_frame(db_uri: str, family: str, manifest: dict[str, object]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="paperrush_positive_history_") as tmp_dir:
        csv_path = Path(tmp_dir) / f"{family.lower()}_positive_history.csv"
        log(f"Exporting {family} positive-row history for amount-stage affinity")
        run_psql_copy(db_uri, build_regressor_positive_query(family), csv_path)
        frame = pd.read_csv(csv_path, parse_dates=["onsaledate", "offsaledate"])
    return assign_split_from_manifest(frame, manifest)


def add_scoring_affinity_features(
    score_frame: pd.DataFrame,
    score_reduced: np.ndarray,
    history_frame: pd.DataFrame,
    history_reduced: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    score_frame = score_frame.copy()
    train_targets = history_frame.loc[history_frame["split"] == "train", "sales_target"].astype(float)
    low_threshold = float(train_targets.quantile(EMBEDDING_AFFINITY_LOW_QUANTILE))
    tail_threshold = float(train_targets.quantile(EMBEDDING_AFFINITY_TAIL_QUANTILE))
    score_vectors = row_normalize(score_reduced[:, :EMBEDDING_AFFINITY_COMPONENTS].astype(np.float32))
    history_vectors = row_normalize(history_reduced[:, :EMBEDDING_AFFINITY_COMPONENTS].astype(np.float32))
    low_mask = history_frame["sales_target"].to_numpy(dtype=np.float32) <= low_threshold
    tail_mask = history_frame["sales_target"].to_numpy(dtype=np.float32) >= tail_threshold
    low_weight = np.ones(len(history_frame), dtype=np.float32)
    tail_weight = history_frame["sales_target"].to_numpy(dtype=np.float32)
    summary: dict[str, object] = {
        "method": "score_rows_against_completed_core_context_tail_low_embedding_centroid_affinity",
        "components": EMBEDDING_AFFINITY_COMPONENTS,
        "low_sales_threshold": low_threshold,
        "tail_sales_threshold": tail_threshold,
        "history_rows": int(len(history_frame)),
        "contexts": {},
    }

    affinity_columns: dict[str, np.ndarray] = {}
    for prefix, group_cols in EMBEDDING_AFFINITY_CONTEXTS.items():
        log(f"Building scoring embedding affinity features for {prefix}")
        tail_history = cumulative_embedding_context(history_frame, history_vectors, group_cols, tail_mask, tail_weight)
        low_history = cumulative_embedding_context(history_frame, history_vectors, group_cols, low_mask, low_weight)
        tail_cols = [col for col in tail_history.columns if col.startswith("_affinity_v") or col in {"_obs", "_sales_sum", "_weight_sum"}]
        low_cols = [col for col in low_history.columns if col.startswith("_affinity_v") or col in {"_obs", "_sales_sum", "_weight_sum"}]
        tail_merged = asof_context_merge(score_frame, tail_history, group_cols, tail_cols)
        low_merged = asof_context_merge(score_frame, low_history, group_cols, low_cols)
        tail_features = context_similarity_features(score_vectors, tail_merged, f"{prefix}_tail")
        low_features = context_similarity_features(score_vectors, low_merged, f"{prefix}_low")
        tail_similarity = tail_features[f"{prefix}_tail_similarity"].to_numpy(dtype=np.float32)
        low_similarity = low_features[f"{prefix}_low_similarity"].to_numpy(dtype=np.float32)
        affinity_columns[f"{prefix}_tail_similarity"] = tail_similarity
        affinity_columns[f"{prefix}_low_similarity"] = low_similarity
        affinity_columns[f"{prefix}_tail_minus_low_similarity"] = (tail_similarity - low_similarity).astype(np.float32)
        affinity_columns[f"{prefix}_tail_obs"] = tail_features[f"{prefix}_tail_obs"].to_numpy(dtype=np.float32)
        affinity_columns[f"{prefix}_low_obs"] = low_features[f"{prefix}_low_obs"].to_numpy(dtype=np.float32)
        affinity_columns[f"{prefix}_tail_avg_sales"] = tail_features[f"{prefix}_tail_avg_sales"].to_numpy(dtype=np.float32)
        affinity_columns[f"{prefix}_low_avg_sales"] = low_features[f"{prefix}_low_avg_sales"].to_numpy(dtype=np.float32)
        summary["contexts"][prefix] = {
            "group_columns": group_cols,
            "tail_history_rows": int(len(tail_history)),
            "low_history_rows": int(len(low_history)),
            "rows_with_tail_context": int((affinity_columns[f"{prefix}_tail_obs"] > 0).sum()),
            "rows_with_low_context": int((affinity_columns[f"{prefix}_low_obs"] > 0).sum()),
        }

    score_frame = pd.concat([score_frame, pd.DataFrame(affinity_columns, index=score_frame.index)], axis=1)
    return score_frame, summary


def add_stage_embedding_features(
    frame: pd.DataFrame,
    db_uri: str,
    family: str,
    dataset_dir: Path,
    stage: str,
    core_products: pd.DataFrame,
    core_matrix: np.ndarray,
    holdout_products: pd.DataFrame,
    holdout_matrix: np.ndarray,
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    family_dir = dataset_dir / family.lower()
    pca_artifact = load_json(family_dir / f"embedding_pca_{stage}.json")
    holdout_reduced = pca_transform_from_artifact(holdout_matrix, pca_artifact)
    holdout_reduced[~holdout_products["has_embedding"].to_numpy(dtype=bool)] = 0.0
    reduced_frame = pd.DataFrame(holdout_reduced, columns=EMBEDDING_FEATURES)
    reduced_frame["product_id"] = holdout_products["product_id"].to_numpy(dtype=np.int64)
    result = frame.merge(reduced_frame, how="left", on="product_id")
    result[EMBEDDING_FEATURES] = result[EMBEDDING_FEATURES].fillna(0.0).astype(np.float32)

    analog_frame, analog_summary = build_holdout_embedding_analog_features(
        core_products,
        core_matrix,
        holdout_products,
        holdout_matrix,
    )
    result = result.merge(analog_frame, how="left", on="product_id")
    result[EMBEDDING_ANALOG_FEATURES] = result[EMBEDDING_ANALOG_FEATURES].fillna(0.0).astype(np.float32)

    if stage == "classifier_all" and (family_dir / "classifier_all.parquet").exists():
        history_cols = [
            "store_id",
            "product_id",
            "onsaledate",
            "offsaledate",
            "split",
            "sales_target",
            "title",
            "segment",
            "subsegment",
            "store_chain",
            "classoftrade",
            *EMBEDDING_FEATURES,
        ]
        log("Loading classifier all-row history for incidence-stage affinity")
        history = pd.read_parquet(family_dir / "classifier_all.parquet", columns=history_cols)
        history["onsaledate"] = pd.to_datetime(history["onsaledate"])
        history["offsaledate"] = pd.to_datetime(history["offsaledate"])
        history_reduced = history[EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    else:
        history = load_positive_history_frame(db_uri, family, manifest)
        core_reduced = pca_transform_from_artifact(core_matrix, pca_artifact)
        product_to_index = {int(pid): idx for idx, pid in enumerate(core_products["product_id"].to_numpy())}
        product_indices = history["product_id"].map(product_to_index)
        history_reduced = np.zeros((len(history), len(EMBEDDING_FEATURES)), dtype=np.float32)
        valid_history = product_indices.notna().to_numpy()
        history_reduced[valid_history] = core_reduced[product_indices[valid_history].astype(np.int64)]

    row_reduced = result[EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    result, affinity_summary = add_scoring_affinity_features(result, row_reduced, history, history_reduced)
    summary = {
        "stage": stage,
        "pca_artifact_path": str(family_dir / f"embedding_pca_{stage}.json"),
        "holdout_row_embedding_coverage": {
            "rows": int(len(frame)),
            "rows_with_embedding": int(frame["product_id"].isin(holdout_products.loc[holdout_products["has_embedding"], "product_id"]).sum()),
            "rows_missing_embedding": int((~frame["product_id"].isin(holdout_products.loc[holdout_products["has_embedding"], "product_id"])).sum()),
        },
        "analog_features": analog_summary,
        "affinity_features": affinity_summary,
    }
    return result, summary


def load_validation_raw_probability_edges(
    dataset_dir: Path,
    classifier_run_dir: Path,
    bins: int,
) -> np.ndarray:
    artifact = incidence_artifact_from_payload(load_json(classifier_run_dir / "encoding_artifact.json"))
    booster = xgb.Booster()
    booster.load_model(classifier_run_dir / "model.json")
    columns = list(dict.fromkeys([*artifact.numeric_features, *artifact.categorical_features, "split", "sales_target"]))
    log("Loading validation rows to reconstruct incidence calibration probability bins")
    frame = pd.read_parquet(dataset_dir / "weeklies" / "classifier_all.parquet", columns=columns)
    valid = frame.loc[frame["split"] == "valid"].copy()
    valid_raw = predict_probability(booster, valid, artifact)
    return probability_bins_from_validation(valid_raw, bins)


def apply_saved_probability_bin_calibration(
    raw_probability: np.ndarray,
    probability_edges: np.ndarray,
    calibration_run_dir: Path,
) -> np.ndarray:
    table = pd.read_csv(calibration_run_dir / "group_multiplier_table_probability_bin.csv")
    frame = pd.DataFrame({"probability_bin": assign_probability_bin(raw_probability, probability_edges).astype(str)})
    return apply_group_multiplier(frame, raw_probability, ["probability_bin"], table, global_multiplier=1.0).astype(np.float32)


def parse_selected_pipeline_variant(variant: str) -> dict[str, float | str]:
    if not variant.startswith("G_soft_gate_amount_"):
        raise ValueError(f"Unsupported selected pipeline variant: {variant}")
    mode = "calibrated" if "_calibrated_" in variant else "raw"
    tokens = variant.split("_")
    payload: dict[str, float | str] = {"mode": mode}
    for token in tokens:
        if token.startswith("t") and len(token) > 1 and token[1].isdigit():
            payload["threshold"] = float(token[1:])
        elif token.startswith("floor"):
            payload["floor"] = float(token.replace("floor", ""))
        elif token.startswith("low") and token != "low":
            payload["low_factor"] = float(token.replace("low", ""))
        elif token.startswith("mid"):
            payload["mid_factor"] = float(token.replace("mid", ""))
        elif token.startswith("top"):
            payload["top_factor"] = float(token.replace("top", ""))
        elif token.startswith("scale"):
            payload["global_scale"] = float(token.replace("scale", ""))
    required = {"threshold", "floor", "low_factor", "mid_factor", "top_factor", "global_scale"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Could not parse {missing} from selected pipeline variant {variant}")
    return payload


def prediction_summary(frame: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in prediction_cols:
        values = frame[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "prediction": col,
                "rows": int(len(values)),
                "predicted_sum": float(values.sum()),
                "prediction_mean": float(values.mean()),
                "prediction_p50": float(np.quantile(values, 0.50)),
                "prediction_p75": float(np.quantile(values, 0.75)),
                "prediction_p90": float(np.quantile(values, 0.90)),
                "prediction_p95": float(np.quantile(values, 0.95)),
                "prediction_p99": float(np.quantile(values, 0.99)),
                "nonzero_rows": int((values > 0).sum()),
                "nonzero_rate": float((values > 0).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def grouped_summary(frame: pd.DataFrame, group_cols: list[str], prediction_col: str) -> pd.DataFrame:
    return (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            rows=(prediction_col, "size"),
            predicted_sum=(prediction_col, "sum"),
            prediction_mean=(prediction_col, "mean"),
            incidence_raw_mean=("incidence_raw_probability", "mean"),
            incidence_calibrated_mean=("incidence_calibrated_probability", "mean"),
            positive_regressor_mean=("positive_regressor_prediction", "mean"),
        )
        .reset_index()
        .sort_values("predicted_sum", ascending=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Weeklies July holdout rows with the selected production benchmark pipeline.")
    parser.add_argument("--family", choices=["Weeklies"], default="Weeklies")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--amount-run-dir", type=Path, default=DEFAULT_AMOUNT_RUN_DIR)
    parser.add_argument("--tail-run-dir", type=Path, default=DEFAULT_TAIL_RUN_DIR)
    parser.add_argument("--classifier-run-dir", type=Path, default=DEFAULT_CLASSIFIER_RUN_DIR)
    parser.add_argument("--calibration-run-dir", type=Path, default=DEFAULT_CALIBRATION_RUN_DIR)
    parser.add_argument("--pipeline-variant", default=DEFAULT_PIPELINE_VARIANT)
    parser.add_argument("--probability-bins", type=int, default=10)
    parser.add_argument("--calibration-mode", choices=["legacy_probability_bin", "none"], default="legacy_probability_bin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    family_key = args.family.lower()
    run_dir = args.output_dir / family_key / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    log("Loading saved model manifests and artifact contracts")
    amount_manifest = load_json(args.dataset_dir / family_key / "manifest_regressor_positive.json")
    classifier_manifest = load_json(args.dataset_dir / family_key / "manifest_classifier_all.json")
    calibration_summary: dict[str, object] = {"selected_variant": "none", "selected_variant_payload": {}}
    if args.calibration_mode == "legacy_probability_bin":
        calibration_summary = load_json(args.calibration_run_dir / "run_summary.json")
        if calibration_summary.get("selected_variant") != "group_probability_bin_target_1.20":
            raise ValueError(f"Unexpected calibration variant: {calibration_summary.get('selected_variant')}")

    frame = load_holdout_frame(args.db_uri, args.family)
    if frame.empty:
        raise ValueError("No Weeklies holdout rows found to score")
    duplicate_rows = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_rows:
        raise ValueError(f"Holdout has {duplicate_rows} duplicate store/product/onsale keys")

    log("Adding completed-history priors from core")
    base_frame, prior_summary = add_completed_prior_features(frame, args.family, args.db_uri)
    core_products, core_matrix = load_family_embeddings(args.db_uri, args.family)
    holdout_products, holdout_matrix, missing_embeddings = load_holdout_embeddings(args.db_uri, args.family)

    log("Building amount-stage holdout features")
    amount_features, amount_embedding_summary = add_stage_embedding_features(
        base_frame,
        args.db_uri,
        args.family,
        args.dataset_dir,
        "regressor_positive",
        core_products,
        core_matrix,
        holdout_products,
        holdout_matrix,
        amount_manifest,
    )
    amount_features = add_derived_numeric_features(amount_features)

    log("Building incidence-stage holdout features")
    incidence_features, incidence_embedding_summary = add_stage_embedding_features(
        base_frame,
        args.db_uri,
        args.family,
        args.dataset_dir,
        "classifier_all",
        core_products,
        core_matrix,
        holdout_products,
        holdout_matrix,
        classifier_manifest,
    )
    incidence_features = add_derived_numeric_features(incidence_features)

    log("Scoring positive amount stack")
    base_booster, base_artifact, base_objective = load_base_regressor(args.amount_run_dir)
    base_prediction = predict_frame(base_booster, amount_features, base_artifact, base_objective).astype(np.float32)
    tail_booster, tail_artifact, tail_summary = load_tail_model(args.tail_run_dir)
    tail_probability = predict_probability(tail_booster, amount_features, tail_artifact).astype(np.float32)
    band_booster = load_optional_band_model(args.tail_run_dir)
    low_booster = load_optional_low_model(args.tail_run_dir)
    band_probability = predict_probability(band_booster, amount_features, tail_artifact).astype(np.float32) if band_booster else None
    low_probability = predict_probability(low_booster, amount_features, tail_artifact).astype(np.float32) if low_booster else None
    positive_prediction = apply_tail_multiplier(
        base_prediction,
        tail_probability,
        tail_summary,
        band_probability=band_probability,
        low_probability=low_probability,
    )

    log("Scoring incidence classifier")
    classifier_artifact = incidence_artifact_from_payload(load_json(args.classifier_run_dir / "encoding_artifact.json"))
    classifier_booster = xgb.Booster()
    classifier_booster.load_model(args.classifier_run_dir / "model.json")
    incidence_raw = predict_probability(classifier_booster, incidence_features, classifier_artifact).astype(np.float32)
    probability_edges: np.ndarray | None = None
    if args.calibration_mode == "legacy_probability_bin":
        probability_edges = load_validation_raw_probability_edges(args.dataset_dir, args.classifier_run_dir, args.probability_bins)
        incidence_calibrated = apply_saved_probability_bin_calibration(incidence_raw, probability_edges, args.calibration_run_dir)
    else:
        incidence_calibrated = incidence_raw.copy()

    variant_config = parse_selected_pipeline_variant(args.pipeline_variant)
    incidence_source = incidence_calibrated if variant_config["mode"] == "calibrated" else incidence_raw
    gate = (incidence_source >= float(variant_config["threshold"])).astype(np.float32)
    incidence_layer = float(variant_config["floor"]) + (1.0 - float(variant_config["floor"])) * gate
    amount_multiplier = amount_rank_bucket_multiplier(
        positive_prediction,
        low_factor=float(variant_config["low_factor"]),
        mid_factor=float(variant_config["mid_factor"]),
        top_factor=float(variant_config["top_factor"]),
        global_scale=float(variant_config["global_scale"]),
    )
    final_prediction = np.clip(positive_prediction.astype(np.float64) * incidence_layer * amount_multiplier, 0.0, None).astype(np.float32)

    output = frame[META_COLUMNS].copy()
    output["incidence_raw_probability"] = incidence_raw
    output["incidence_calibrated_probability"] = incidence_calibrated
    output["incidence_gate"] = gate
    output["positive_base_prediction"] = base_prediction
    output["tail_probability"] = tail_probability
    output["low_probability"] = low_probability if low_probability is not None else np.nan
    output["positive_regressor_prediction"] = positive_prediction
    output["amount_rank"] = probability_rank(positive_prediction).astype(np.float32)
    output["amount_multiplier"] = amount_multiplier
    output["july_holdout_prediction"] = final_prediction

    predictions_csv = run_dir / "july_holdout_weeklies_predictions.csv"
    predictions_parquet = run_dir / "july_holdout_weeklies_predictions.parquet"
    output.to_csv(predictions_csv, index=False)
    output.to_parquet(predictions_parquet, index=False)
    prediction_summary(output, ["positive_regressor_prediction", "july_holdout_prediction"]).to_csv(
        run_dir / "prediction_summary.csv",
        index=False,
    )
    grouped_summary(output, ["onsaledate"], "july_holdout_prediction").to_csv(run_dir / "summary_by_onsaledate.csv", index=False)
    grouped_summary(output, ["title"], "july_holdout_prediction").to_csv(run_dir / "summary_by_title.csv", index=False)
    grouped_summary(output, ["classoftrade"], "july_holdout_prediction").to_csv(run_dir / "summary_by_classoftrade.csv", index=False)
    grouped_summary(output, ["store_chain"], "july_holdout_prediction").to_csv(run_dir / "summary_by_store_chain.csv", index=False)
    if not missing_embeddings.empty:
        missing_embeddings.to_csv(run_dir / "missing_holdout_embeddings.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "run_dir": str(run_dir),
        "row_count": int(len(output)),
        "product_count": int(output["product_id"].nunique()),
        "store_count": int(output["store_id"].nunique()),
        "onsaledate_min": output["onsaledate"].min().strftime("%Y-%m-%d"),
        "onsaledate_max": output["onsaledate"].max().strftime("%Y-%m-%d"),
        "pipeline_variant": args.pipeline_variant,
        "pipeline_variant_config": variant_config,
        "artifact_paths": {
            "amount_run_dir": str(args.amount_run_dir),
            "tail_run_dir": str(args.tail_run_dir),
            "classifier_run_dir": str(args.classifier_run_dir),
            "calibration_run_dir": str(args.calibration_run_dir),
            "dataset_dir": str(args.dataset_dir),
        },
        "tail_multiplier": tail_multiplier_from_summary(tail_summary),
        "calibration_selected_variant": calibration_summary.get("selected_variant"),
        "calibration_selected_variant_payload": calibration_summary.get("selected_variant_payload"),
        "calibration_mode": args.calibration_mode,
        "probability_bin_edges": (
            [float(value) if math.isfinite(float(value)) else str(value) for value in probability_edges]
            if probability_edges is not None
            else []
        ),
        "holdout_embedding_missing_products": missing_embeddings.to_dict(orient="records"),
        "prior_features": prior_summary,
        "amount_embedding_features": amount_embedding_summary,
        "incidence_embedding_features": incidence_embedding_summary,
        "outputs": {
            "predictions_csv": str(predictions_csv),
            "predictions_parquet": str(predictions_parquet),
            "prediction_summary": str(run_dir / "prediction_summary.csv"),
            "summary_by_onsaledate": str(run_dir / "summary_by_onsaledate.csv"),
            "summary_by_title": str(run_dir / "summary_by_title.csv"),
            "summary_by_classoftrade": str(run_dir / "summary_by_classoftrade.csv"),
            "summary_by_store_chain": str(run_dir / "summary_by_store_chain.csv"),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote July holdout scoring outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
