#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = NEW_DIR / "output" / "modeling_datasets"
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "model_runs"
UNKNOWN_TOKEN = "__UNKNOWN__"
MISSING_TOKEN = "UNKNOWN"
WEIGHTING_SCHEMES: dict[str, dict[str, float]] = {
    "uniform": {
        "base": 1.0,
        "bottom_30_pct": 1.0,
        "top_30_pct": 1.0,
        "top_10_pct": 1.0,
    },
    "tail_focus": {
        "base": 0.7,
        "bottom_30_pct": 1.5,
        "top_30_pct": 2.0,
        "top_10_pct": 1.0,
    },
    "extreme_focus": {
        "base": 1.0,
        "bottom_30_pct": 5.0,
        "top_30_pct": 1.0,
        "top_10_pct": 3.0,
    },
    "category_balance": {
        "base": 1.0,
        "balance_power": 0.5,
        "balance_min": 0.5,
        "balance_max": 2.0,
        "bottom_30_pct": 1.0,
        "top_20_pct": 1.0,
        "normalize_mean": 1.0,
    },
    "category_tail_balance": {
        "base": 1.0,
        "balance_power": 0.5,
        "balance_min": 0.5,
        "balance_max": 2.0,
        "tail_min_group_rows": 100.0,
        "bottom_30_pct": 0.75,
        "top_20_pct": 3.0,
        "normalize_mean": 1.0,
    },
    "category_xtrade_balance": {
        "base": 1.0,
        "balance_power": 0.5,
        "balance_min": 0.5,
        "balance_max": 2.0,
        "bottom_30_pct": 1.0,
        "top_20_pct": 1.0,
        "normalize_mean": 1.0,
    },
    "category_xtrade_tail_balance": {
        "base": 1.0,
        "balance_power": 0.5,
        "balance_min": 0.5,
        "balance_max": 2.0,
        "tail_min_group_rows": 100.0,
        "bottom_30_pct": 0.75,
        "top_20_pct": 3.0,
        "normalize_mean": 1.0,
    },
}
GLOBAL_TAIL_WEIGHTING_SCHEMES = {"tail_focus", "extreme_focus"}
CATEGORY_WEIGHTING_SCHEMES = {
    "category_balance",
    "category_tail_balance",
    "category_xtrade_balance",
    "category_xtrade_tail_balance",
}
QUANTILE_OBJECTIVES: dict[str, float] = {
    "quantile_50": 0.50,
    "quantile_80": 0.80,
    "quantile_90": 0.90,
}
CURVE_LOG1P_OBJECTIVE = "curve_log1p"
ASYM_CURVE_LOG1P_OBJECTIVE = "asym_curve_log1p"
OBJECTIVE_CHOICES = [
    "log1p_squarederror",
    CURVE_LOG1P_OBJECTIVE,
    ASYM_CURVE_LOG1P_OBJECTIVE,
    "poisson",
    "tweedie",
    *QUANTILE_OBJECTIVES.keys(),
]
CURVE_OBJECTIVES = {CURVE_LOG1P_OBJECTIVE, ASYM_CURVE_LOG1P_OBJECTIVE}
LOG_UNIT_OBJECTIVES = {"log1p_squarederror", *CURVE_OBJECTIVES}
CURVE_DECILES = 10
DEFAULT_CURVE_PENALTY = 0.15
DEFAULT_CURVE_BOTTOM_WEIGHT = 1.5
DEFAULT_CURVE_MIDDLE_WEIGHT = 1.0
DEFAULT_CURVE_TOP_WEIGHT = 2.0
DEFAULT_ASYM_BOTTOM_OVER_WEIGHT = 3.0
DEFAULT_ASYM_BOTTOM_UNDER_WEIGHT = 0.35
DEFAULT_ASYM_MIDDLE_WEIGHT = 1.0
DEFAULT_ASYM_TOP_UNDER_WEIGHT = 3.0
DEFAULT_ASYM_TOP_OVER_WEIGHT = 0.6
CURVE_LOG_PRED_MIN = -10.0
CURVE_LOG_PRED_MAX = 10.0

# Current intentionally excluded dataset columns plus a quick feature ablation
# switchboard. Add exact active feature names here and rerun one family/objective
# without rebuilding the Parquet modeling dataset.
COMMON_DROPPED_FEATURES: list[str] = [
    "product_id",
    # "store_id",
    "postal_code",
    "barcode",

    "onsaledate",
    "type",
    "sales_target",
    "soldqty_raw",
    "drawqty",
    "positive_sale_flag",
    "negative_sales_flag",
    "stockout_proxy_flag",
    "split",
]

FAMILY_DROPPED_FEATURES: dict[str, list[str]] = {
    "Weeklies": [
        *COMMON_DROPPED_FEATURES,
        # "store_title_prior_positive_avg_sales",
        # "store_title_prior_obs",
    ],
    "SIP": [
        *COMMON_DROPPED_FEATURES,
        "title",
        # "store_segment_prior_positive_avg_sales",
        # "store_subsegment_prior_positive_avg_sales",
    ],
}


@dataclass
class EncodingArtifact:
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_medians: dict[str, float]
    category_maps: dict[str, dict[str, int]]
    category_sizes: dict[str, int]
    feature_names: list[str]


@dataclass
class EncodedFrame:
    matrix: sparse.csr_matrix
    labels: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class CurveObjectiveConfig:
    penalty: float = DEFAULT_CURVE_PENALTY
    bottom_weight: float = DEFAULT_CURVE_BOTTOM_WEIGHT
    middle_weight: float = DEFAULT_CURVE_MIDDLE_WEIGHT
    top_weight: float = DEFAULT_CURVE_TOP_WEIGHT
    asym_bottom_over_weight: float = DEFAULT_ASYM_BOTTOM_OVER_WEIGHT
    asym_bottom_under_weight: float = DEFAULT_ASYM_BOTTOM_UNDER_WEIGHT
    asym_middle_weight: float = DEFAULT_ASYM_MIDDLE_WEIGHT
    asym_top_under_weight: float = DEFAULT_ASYM_TOP_UNDER_WEIGHT
    asym_top_over_weight: float = DEFAULT_ASYM_TOP_OVER_WEIGHT
    deciles: int = CURVE_DECILES

    def decile_weights(self) -> np.ndarray:
        weights = np.full(self.deciles, self.middle_weight, dtype=np.float32)
        if self.deciles >= 4:
            weights[:4] = self.bottom_weight
        if self.deciles >= 2:
            weights[-2:] = self.top_weight
        return weights

    def asymmetric_decile_weights(self, log_ratio: np.ndarray) -> np.ndarray:
        weights = np.full(self.deciles, self.asym_middle_weight, dtype=np.float64)
        over_prediction = log_ratio >= 0.0
        bottom = np.arange(self.deciles) < min(4, self.deciles)
        top = np.arange(self.deciles) >= max(0, self.deciles - 2)
        weights[bottom & over_prediction] = self.asym_bottom_over_weight
        weights[bottom & ~over_prediction] = self.asym_bottom_under_weight
        weights[top & ~over_prediction] = self.asym_top_under_weight
        weights[top & over_prediction] = self.asym_top_over_weight
        return weights

    def to_dict(self) -> dict[str, object]:
        return {
            "penalty": self.penalty,
            "bottom_weight": self.bottom_weight,
            "middle_weight": self.middle_weight,
            "top_weight": self.top_weight,
            "asym_bottom_over_weight": self.asym_bottom_over_weight,
            "asym_bottom_under_weight": self.asym_bottom_under_weight,
            "asym_middle_weight": self.asym_middle_weight,
            "asym_top_under_weight": self.asym_top_under_weight,
            "asym_top_over_weight": self.asym_top_over_weight,
            "deciles": self.deciles,
            "prediction_log_clip_min": CURVE_LOG_PRED_MIN,
            "prediction_log_clip_max": CURVE_LOG_PRED_MAX,
        }


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom == 0:
        return math.nan
    return float(np.abs(y_true - y_pred).sum() / denom)


def unit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(y_true.sum())
    if denom == 0:
        return math.nan
    return float(y_pred.sum() / denom)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, object]:
    return {
        "slice": label,
        "rows": int(len(y_true)),
        "actual_sum": float(y_true.sum()),
        "predicted_sum": float(y_pred.sum()),
        "unit_ratio_pred_over_actual": unit_ratio(y_true, y_pred),
        "bias_pred_minus_actual": float(y_pred.sum() - y_true.sum()),
        "actual_mean": float(y_true.mean()) if len(y_true) else math.nan,
        "prediction_mean": float(y_pred.mean()) if len(y_pred) else math.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else math.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else math.nan,
        "wape": safe_wape(y_true, y_pred),
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_dir: Path, family: str, stage: str) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    family_dir = dataset_dir / family.lower()
    parquet_path = family_dir / f"{stage}.parquet"
    manifest_path = family_dir / f"manifest_{stage}.json"
    contract_path = family_dir / f"feature_contract_{stage}.json"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing dataset: {parquet_path}")
    frame = pd.read_parquet(parquet_path)
    frame["onsaledate"] = pd.to_datetime(frame["onsaledate"])
    return frame, load_json(manifest_path), load_json(contract_path)


def apply_feature_drops(contract: dict[str, object], family: str) -> tuple[dict[str, object], dict[str, object]]:
    numeric_features = list(contract["numeric_features"])
    categorical_features = list(contract["categorical_features"])
    configured_drops = list(dict.fromkeys(FAMILY_DROPPED_FEATURES.get(family, [])))
    all_features = set(numeric_features) | set(categorical_features)
    active_drop_set = set(configured_drops) & all_features
    effective_contract = dict(contract)
    effective_contract["numeric_features"] = [col for col in numeric_features if col not in active_drop_set]
    effective_contract["categorical_features"] = [col for col in categorical_features if col not in active_drop_set]
    if not effective_contract["numeric_features"] and not effective_contract["categorical_features"]:
        raise ValueError(f"All features were dropped for {family}; keep at least one feature.")
    drop_report = {
        "family": family,
        "configured_dropped_features": configured_drops,
        "active_dropped_numeric_features": [col for col in numeric_features if col in active_drop_set],
        "active_dropped_categorical_features": [col for col in categorical_features if col in active_drop_set],
        "configured_non_contract_columns": [col for col in configured_drops if col not in all_features],
        "numeric_features_before": len(numeric_features),
        "categorical_features_before": len(categorical_features),
        "numeric_features_after": len(effective_contract["numeric_features"]),
        "categorical_features_after": len(effective_contract["categorical_features"]),
    }
    return effective_contract, drop_report


def fit_encoder(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> EncodingArtifact:
    numeric_medians = {}
    for col in numeric_features:
        value = frame[col].median()
        numeric_medians[col] = float(value) if not pd.isna(value) else 0.0

    category_maps: dict[str, dict[str, int]] = {}
    category_sizes: dict[str, int] = {}
    categorical_feature_names = []
    for col in categorical_features:
        values = frame[col].astype("string").fillna(MISSING_TOKEN).astype(str)
        levels = sorted(set(values))
        if UNKNOWN_TOKEN not in levels:
            levels.append(UNKNOWN_TOKEN)
        mapping = {level: idx for idx, level in enumerate(levels)}
        category_maps[col] = mapping
        category_sizes[col] = len(mapping)
        categorical_feature_names.extend([f"{col}={level}" for level in levels])

    return EncodingArtifact(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_medians=numeric_medians,
        category_maps=category_maps,
        category_sizes=category_sizes,
        feature_names=[*numeric_features, *categorical_feature_names],
    )


def category_weight_columns(frame: pd.DataFrame, weighting: str) -> list[str]:
    if "title" in frame.columns and frame["title"].nunique(dropna=False) > 1:
        columns = ["title"]
    else:
        columns = ["subsegment"]
    if weighting in {"category_xtrade_balance", "category_xtrade_tail_balance"}:
        columns.append("classoftrade")
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Category weighting needs missing columns: {missing}")
    return columns


def normalize_mean_one(weights: np.ndarray) -> np.ndarray:
    mean = float(weights.mean()) if len(weights) else 1.0
    if mean > 0:
        weights = weights / mean
    return weights.astype(np.float32, copy=False)


def category_balanced_weights(frame: pd.DataFrame, weighting: str) -> np.ndarray:
    scheme = WEIGHTING_SCHEMES[weighting]
    weights = np.full(len(frame), scheme["base"], dtype=np.float32)
    if len(frame) == 0:
        return weights

    group_cols = category_weight_columns(frame, weighting)
    group_frame = frame[group_cols].astype("string").fillna(MISSING_TOKEN).astype(str)
    group_counts = group_frame.groupby(group_cols, dropna=False)[group_cols[0]].transform("size").to_numpy(dtype=np.float32)
    median_count = float(np.median(group_counts))
    balance_factor = np.power(median_count / np.maximum(group_counts, 1.0), scheme["balance_power"])
    balance_factor = np.clip(balance_factor, scheme["balance_min"], scheme["balance_max"])
    weights *= balance_factor.astype(np.float32)

    if weighting in {"category_tail_balance", "category_xtrade_tail_balance"}:
        percentile_rank = frame.groupby(group_cols, dropna=False)["sales_target"].rank(method="first", pct=True).to_numpy(dtype=np.float32)
        tail_eligible = group_counts >= scheme["tail_min_group_rows"]
        weights[(percentile_rank <= 0.30) & tail_eligible] *= scheme["bottom_30_pct"]
        weights[(percentile_rank > 0.80) & tail_eligible] *= scheme["top_20_pct"]

    if scheme.get("normalize_mean", 0.0):
        weights = normalize_mean_one(weights)
    return weights


def sample_weights(frame: pd.DataFrame, weighting: str) -> np.ndarray:
    if weighting not in WEIGHTING_SCHEMES:
        raise ValueError(f"Unsupported weighting scheme: {weighting}")
    if weighting in CATEGORY_WEIGHTING_SCHEMES:
        return category_balanced_weights(frame, weighting)
    scheme = WEIGHTING_SCHEMES[weighting]
    weights = np.full(len(frame), scheme["base"], dtype=np.float32)
    if weighting == "uniform" or len(frame) == 0:
        return weights
    if weighting not in GLOBAL_TAIL_WEIGHTING_SCHEMES:
        raise ValueError(f"Unsupported weighting scheme: {weighting}")
    percentile_rank = frame["sales_target"].rank(method="first", pct=True).to_numpy(dtype=np.float32)
    weights[percentile_rank <= 0.30] = scheme["bottom_30_pct"]
    weights[percentile_rank > 0.70] = scheme["top_30_pct"]
    weights[percentile_rank > 0.90] = scheme["top_10_pct"]
    return weights


def weight_report(frame: pd.DataFrame, weighting: str) -> dict[str, object]:
    weights = sample_weights(frame, weighting)
    report = {
        "scheme": weighting,
        "parameters": WEIGHTING_SCHEMES[weighting],
        "rows": int(len(frame)),
        "weight_sum": float(weights.sum()),
        "weight_mean": float(weights.mean()) if len(weights) else math.nan,
        "weight_min": float(weights.min()) if len(weights) else math.nan,
        "weight_max": float(weights.max()) if len(weights) else math.nan,
    }
    if weighting in CATEGORY_WEIGHTING_SCHEMES and len(frame):
        group_cols = category_weight_columns(frame, weighting)
        group_sizes = frame.groupby(group_cols, dropna=False).size()
        report.update(
            {
                "category_weight_columns": group_cols,
                "category_groups": int(len(group_sizes)),
                "category_group_rows_min": int(group_sizes.min()),
                "category_group_rows_median": float(group_sizes.median()),
                "category_group_rows_max": int(group_sizes.max()),
            }
        )
    return report


def transform_frame(frame: pd.DataFrame, artifact: EncodingArtifact, objective: str, weighting: str) -> EncodedFrame:
    numeric_parts = []
    if artifact.numeric_features:
        numeric = frame[artifact.numeric_features].copy()
        for col, value in artifact.numeric_medians.items():
            numeric[col] = numeric[col].fillna(value)
        numeric_parts.append(sparse.csr_matrix(numeric.to_numpy(dtype=np.float32)))

    row_count = len(frame)
    cat_total_cols = sum(artifact.category_sizes.values())
    if cat_total_cols:
        rows = []
        cols = []
        data = []
        offset = 0
        row_indices = np.arange(row_count, dtype=np.int32)
        for col in artifact.categorical_features:
            mapping = artifact.category_maps[col]
            unknown_idx = mapping[UNKNOWN_TOKEN]
            values = frame[col].astype("string").fillna(MISSING_TOKEN).astype(str)
            codes = values.map(mapping).fillna(unknown_idx).to_numpy(dtype=np.int32)
            rows.append(row_indices)
            cols.append(codes + offset)
            data.append(np.ones(row_count, dtype=np.float32))
            offset += artifact.category_sizes[col]
        cat_matrix = sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row_count, cat_total_cols),
        )
        numeric_parts.append(cat_matrix)

    matrix = sparse.hstack(numeric_parts, format="csr") if len(numeric_parts) > 1 else numeric_parts[0]
    target = frame["sales_target"].to_numpy(dtype=np.float32)
    labels = np.log1p(target) if objective in LOG_UNIT_OBJECTIVES else target
    weights = sample_weights(frame, weighting)
    return EncodedFrame(matrix=matrix, labels=labels, weights=weights)


def artifact_payload(artifact: EncodingArtifact) -> dict[str, object]:
    return {
        "unknown_token": UNKNOWN_TOKEN,
        "missing_token": MISSING_TOKEN,
        "numeric_features": artifact.numeric_features,
        "categorical_features": artifact.categorical_features,
        "numeric_medians": artifact.numeric_medians,
        "category_maps": artifact.category_maps,
        "category_sizes": artifact.category_sizes,
        "feature_names": artifact.feature_names,
    }


def xgb_params(objective: str, seed: int, base_score: float | None = None) -> dict[str, object]:
    base = {
        "tree_method": "hist",
        "max_depth": 6,
        "min_child_weight": 25,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "seed": seed,
        "eval_metric": "rmse",
    }
    if base_score is not None:
        base["base_score"] = float(base_score)
    if objective == "log1p_squarederror":
        return {**base, "objective": "reg:squarederror"}
    if objective in CURVE_OBJECTIVES:
        return {**base, "objective": "reg:squarederror"}
    if objective == "poisson":
        return {**base, "objective": "count:poisson", "max_delta_step": 1.0}
    if objective == "tweedie":
        return {**base, "objective": "reg:tweedie", "tweedie_variance_power": 1.3}
    if objective in QUANTILE_OBJECTIVES:
        return {
            **base,
            "objective": "reg:quantileerror",
            "quantile_alpha": QUANTILE_OBJECTIVES[objective],
            "eval_metric": "quantile",
        }
    raise ValueError(f"Unsupported objective: {objective}")


def inverse_predictions(raw_pred: np.ndarray, objective: str) -> np.ndarray:
    if objective in LOG_UNIT_OBJECTIVES:
        clipped = np.clip(raw_pred, CURVE_LOG_PRED_MIN, CURVE_LOG_PRED_MAX)
        return np.clip(np.expm1(clipped), 0.0, None)
    return np.clip(raw_pred, 0.0, None)


def curve_decile_cache(labels: np.ndarray, config: CurveObjectiveConfig) -> dict[str, np.ndarray]:
    actual_units = np.clip(np.expm1(labels.astype(np.float64)), 0.0, None).astype(np.float32)
    row_count = len(actual_units)
    if row_count == 0:
        empty_float = np.array([], dtype=np.float32)
        empty_int = np.array([], dtype=np.int32)
        return {
            "actual_units": empty_float,
            "decile_id": empty_int,
            "actual_sums": empty_float,
            "counts": empty_float,
            "decile_weights": empty_float,
        }

    order = np.lexsort((np.arange(row_count), actual_units))
    ranks = np.empty(row_count, dtype=np.int32)
    ranks[order] = np.arange(row_count, dtype=np.int32)
    decile_id = np.floor(ranks * config.deciles / row_count).astype(np.int32)
    decile_id = np.clip(decile_id, 0, config.deciles - 1)
    actual_sums = np.bincount(decile_id, weights=actual_units, minlength=config.deciles).astype(np.float32)
    counts = np.bincount(decile_id, minlength=config.deciles).astype(np.float32)
    decile_weights = config.decile_weights()
    return {
        "actual_units": actual_units,
        "decile_id": decile_id,
        "actual_sums": actual_sums,
        "counts": counts,
        "decile_weights": decile_weights,
    }


def curve_log1p_objective(config: CurveObjectiveConfig):
    cache_by_matrix: dict[int, dict[str, np.ndarray]] = {}

    def objective(raw_pred: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        key = id(dmatrix)
        if key not in cache_by_matrix:
            cache_by_matrix[key] = curve_decile_cache(dmatrix.get_label(), config)
        cache = cache_by_matrix[key]
        labels = dmatrix.get_label().astype(np.float32)
        row_weights = dmatrix.get_weight()
        if len(row_weights) == 0:
            row_weights = np.ones_like(labels, dtype=np.float32)
        raw_clipped = np.clip(raw_pred.astype(np.float64), CURVE_LOG_PRED_MIN, CURVE_LOG_PRED_MAX)
        pred_units = np.expm1(raw_clipped)
        unit_derivative = np.exp(raw_clipped)

        grad = row_weights.astype(np.float64) * (raw_clipped - labels)
        hess = row_weights.astype(np.float64)

        if config.penalty > 0 and len(labels):
            decile_id = cache["decile_id"]
            actual_sums = cache["actual_sums"].astype(np.float64)
            counts = cache["counts"].astype(np.float64)
            decile_weights = cache["decile_weights"].astype(np.float64)
            pred_sums = np.bincount(decile_id, weights=pred_units, minlength=config.deciles).astype(np.float64)
            safe_actual = np.maximum(actual_sums, 1e-6)
            ratio_error = (pred_sums / safe_actual) - 1.0
            scale = 2.0 * config.penalty * decile_weights[decile_id] * counts[decile_id]
            decile_grad = scale * ratio_error[decile_id] * unit_derivative / safe_actual[decile_id]
            decile_hess = scale * (unit_derivative * unit_derivative) / (safe_actual[decile_id] * safe_actual[decile_id])
            grad += decile_grad
            hess += decile_hess

        hess = np.maximum(hess, 1e-6)
        return grad.astype(np.float32), hess.astype(np.float32)

    return objective


def curve_log1p_metric(config: CurveObjectiveConfig):
    cache_by_matrix: dict[int, dict[str, np.ndarray]] = {}

    def metric(raw_pred: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
        key = id(dmatrix)
        if key not in cache_by_matrix:
            cache_by_matrix[key] = curve_decile_cache(dmatrix.get_label(), config)
        cache = cache_by_matrix[key]
        labels = dmatrix.get_label().astype(np.float32)
        raw_clipped = np.clip(raw_pred.astype(np.float64), CURVE_LOG_PRED_MIN, CURVE_LOG_PRED_MAX)
        log_rmse = float(np.sqrt(np.mean(np.square(raw_clipped - labels)))) if len(labels) else math.nan
        if not len(labels):
            return "curve_log1p_score", log_rmse
        pred_units = np.expm1(raw_clipped)
        decile_id = cache["decile_id"]
        actual_sums = cache["actual_sums"].astype(np.float64)
        decile_weights = cache["decile_weights"].astype(np.float64)
        pred_sums = np.bincount(decile_id, weights=pred_units, minlength=config.deciles).astype(np.float64)
        safe_actual = np.maximum(actual_sums, 1e-6)
        ratio_abs_error = np.abs((pred_sums / safe_actual) - 1.0)
        weighted_curve_error = float(np.average(ratio_abs_error, weights=decile_weights))
        return "curve_log1p_score", log_rmse + config.penalty * weighted_curve_error

    return metric


def asym_curve_log1p_objective(config: CurveObjectiveConfig):
    cache_by_matrix: dict[int, dict[str, np.ndarray]] = {}

    def objective(raw_pred: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        key = id(dmatrix)
        if key not in cache_by_matrix:
            cache_by_matrix[key] = curve_decile_cache(dmatrix.get_label(), config)
        cache = cache_by_matrix[key]
        labels = dmatrix.get_label().astype(np.float32)
        row_weights = dmatrix.get_weight()
        if len(row_weights) == 0:
            row_weights = np.ones_like(labels, dtype=np.float32)
        raw_clipped = np.clip(raw_pred.astype(np.float64), CURVE_LOG_PRED_MIN, CURVE_LOG_PRED_MAX)
        pred_units = np.expm1(raw_clipped)
        unit_derivative = np.exp(raw_clipped)

        grad = row_weights.astype(np.float64) * (raw_clipped - labels)
        hess = row_weights.astype(np.float64)

        if config.penalty > 0 and len(labels):
            decile_id = cache["decile_id"]
            actual_sums = cache["actual_sums"].astype(np.float64)
            counts = cache["counts"].astype(np.float64)
            pred_sums = np.bincount(decile_id, weights=pred_units, minlength=config.deciles).astype(np.float64)
            safe_actual = np.maximum(actual_sums, 1e-6)
            safe_pred = np.maximum(pred_sums, 1e-6)
            log_ratio = np.log(safe_pred / safe_actual)
            direction_weights = config.asymmetric_decile_weights(log_ratio)
            scale = 2.0 * config.penalty * direction_weights[decile_id] * counts[decile_id]
            decile_grad = scale * log_ratio[decile_id] * unit_derivative / safe_pred[decile_id]
            decile_hess = scale * (unit_derivative * unit_derivative) / (safe_pred[decile_id] * safe_pred[decile_id])
            grad += decile_grad
            hess += decile_hess

        hess = np.maximum(hess, 1e-6)
        return grad.astype(np.float32), hess.astype(np.float32)

    return objective


def asym_curve_log1p_metric(config: CurveObjectiveConfig):
    cache_by_matrix: dict[int, dict[str, np.ndarray]] = {}

    def metric(raw_pred: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
        key = id(dmatrix)
        if key not in cache_by_matrix:
            cache_by_matrix[key] = curve_decile_cache(dmatrix.get_label(), config)
        cache = cache_by_matrix[key]
        labels = dmatrix.get_label().astype(np.float32)
        raw_clipped = np.clip(raw_pred.astype(np.float64), CURVE_LOG_PRED_MIN, CURVE_LOG_PRED_MAX)
        log_rmse = float(np.sqrt(np.mean(np.square(raw_clipped - labels)))) if len(labels) else math.nan
        if not len(labels):
            return "asym_curve_log1p_score", log_rmse
        pred_units = np.expm1(raw_clipped)
        decile_id = cache["decile_id"]
        actual_sums = cache["actual_sums"].astype(np.float64)
        counts = cache["counts"].astype(np.float64)
        pred_sums = np.bincount(decile_id, weights=pred_units, minlength=config.deciles).astype(np.float64)
        safe_actual = np.maximum(actual_sums, 1e-6)
        safe_pred = np.maximum(pred_sums, 1e-6)
        log_ratio = np.log(safe_pred / safe_actual)
        direction_weights = config.asymmetric_decile_weights(log_ratio)
        weighted_curve_error = float(np.average(np.abs(log_ratio), weights=direction_weights * np.maximum(counts, 1.0)))
        return "asym_curve_log1p_score", log_rmse + config.penalty * weighted_curve_error

    return metric


def train_booster(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    artifact: EncodingArtifact,
    objective: str,
    rounds: int,
    early_stopping: int,
    seed: int,
    weighting: str,
    label: str,
    curve_config: CurveObjectiveConfig,
) -> tuple[xgb.Booster, np.ndarray]:
    train_encoded = transform_frame(train_df, artifact, objective, weighting)
    valid_encoded = transform_frame(valid_df, artifact, objective, weighting)
    dtrain = xgb.DMatrix(
        train_encoded.matrix,
        label=train_encoded.labels,
        weight=train_encoded.weights,
        feature_names=artifact.feature_names,
    )
    dvalid = xgb.DMatrix(
        valid_encoded.matrix,
        label=valid_encoded.labels,
        weight=valid_encoded.weights,
        feature_names=artifact.feature_names,
    )
    log(f"Training {label}: {len(train_df)} train rows, {len(valid_df)} validation rows")
    custom_objective = None
    custom_metric = None
    base_score = None
    if objective == CURVE_LOG1P_OBJECTIVE:
        custom_objective = curve_log1p_objective(curve_config)
        custom_metric = curve_log1p_metric(curve_config)
        base_score = float(np.mean(train_encoded.labels)) if len(train_encoded.labels) else 0.0
    elif objective == ASYM_CURVE_LOG1P_OBJECTIVE:
        custom_objective = asym_curve_log1p_objective(curve_config)
        custom_metric = asym_curve_log1p_metric(curve_config)
        base_score = float(np.mean(train_encoded.labels)) if len(train_encoded.labels) else 0.0
    booster = xgb.train(
        params=xgb_params(objective, seed, base_score=base_score),
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        obj=custom_objective,
        custom_metric=custom_metric,
        maximize=False,
        early_stopping_rounds=early_stopping,
        verbose_eval=25,
    )
    raw_pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    return booster, inverse_predictions(raw_pred, objective)


def predict_frame(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact, objective: str) -> np.ndarray:
    encoded = transform_frame(frame, artifact, objective, "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    raw = booster.predict(dmatrix, iteration_range=(0, booster.best_iteration + 1))
    return inverse_predictions(raw, objective)


def actual_decile_metrics(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    work = frame.copy()
    work["actual_decile"] = pd.qcut(
        work["sales_target"].rank(method="first"),
        q=min(10, len(work)),
        labels=False,
        duplicates="drop",
    )
    rows = []
    for decile, group in work.groupby("actual_decile", observed=True):
        y_true = group["sales_target"].to_numpy(dtype=np.float32)
        y_pred = group[prediction_col].to_numpy(dtype=np.float32)
        row = metric_row(y_true, y_pred, f"actual_decile_{int(decile)}")
        row["actual_decile"] = int(decile)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def decile_unit_table(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    metrics = actual_decile_metrics(frame, prediction_col).sort_values("actual_decile")
    labels = [f"D{int(decile) + 1}" for decile in metrics["actual_decile"]]
    table = pd.DataFrame(
        [
            metrics["actual_sum"].to_numpy(dtype=np.float64),
            metrics["predicted_sum"].to_numpy(dtype=np.float64),
        ],
        index=["true_sold_units", "pred_sold_units"],
        columns=labels,
    )
    table["Total"] = table.sum(axis=1)
    return table


def log_decile_unit_table(frame: pd.DataFrame, prediction_col: str, label: str) -> pd.DataFrame:
    table = decile_unit_table(frame, prediction_col)
    display = table.round(0).astype("int64")
    log(f"{label} sold-unit sums by actual positive-sales decile")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(display.to_string(), flush=True)
    return table


def issue_date_bias(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    rows = []
    for onsaledate, group in frame.groupby("onsaledate", observed=True):
        row = metric_row(
            group["sales_target"].to_numpy(dtype=np.float32),
            group[prediction_col].to_numpy(dtype=np.float32),
            "issue_date",
        )
        row["onsaledate"] = pd.Timestamp(onsaledate).strftime("%Y-%m-%d")
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("onsaledate")


def calibration_by_prediction_bin(frame: pd.DataFrame, prediction_col: str, bins: int = 10) -> pd.DataFrame:
    work = frame.copy()
    work["prediction_bin"] = pd.qcut(
        work[prediction_col].rank(method="first"),
        q=min(bins, len(work)),
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bin_id, group in work.groupby("prediction_bin", observed=True):
        y_true = group["sales_target"].to_numpy(dtype=np.float32)
        y_pred = group[prediction_col].to_numpy(dtype=np.float32)
        row = metric_row(y_true, y_pred, f"prediction_bin_{int(bin_id)}")
        row["prediction_bin"] = int(bin_id)
        row["actual_p50"] = float(group["sales_target"].quantile(0.50))
        row["actual_p90"] = float(group["sales_target"].quantile(0.90))
        row["prediction_p50"] = float(group[prediction_col].quantile(0.50))
        row["prediction_p90"] = float(group[prediction_col].quantile(0.90))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def tail_capture(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    n = max(1, math.ceil(len(frame) * 0.10))
    actual_top_idx = set(frame.nlargest(n, "sales_target").index)
    pred_top_idx = set(frame.nlargest(n, prediction_col).index)
    overlap_idx = actual_top_idx & pred_top_idx
    y_true = frame["sales_target"].to_numpy(dtype=np.float32)
    y_pred = frame[prediction_col].to_numpy(dtype=np.float32)
    actual_top_actual_sum = float(frame.loc[list(actual_top_idx), "sales_target"].sum())
    pred_top_actual_sum = float(frame.loc[list(pred_top_idx), "sales_target"].sum())
    return pd.DataFrame.from_records(
        [
            {
                "rows": int(len(frame)),
                "top_k_rows": int(n),
                "actual_top10_actual_units": actual_top_actual_sum,
                "predicted_top10_actual_units": pred_top_actual_sum,
                "overlap_rows": int(len(overlap_idx)),
                "overlap_rate_vs_actual_top10": len(overlap_idx) / n,
                "actual_top10_unit_recall_by_predicted_top10": pred_top_actual_sum / actual_top_actual_sum if actual_top_actual_sum else math.nan,
                "predicted_sum_total": float(y_pred.sum()),
                "actual_sum_total": float(y_true.sum()),
                "unit_ratio_pred_over_actual": unit_ratio(y_true, y_pred),
            }
        ]
    )


def actual_tail_priority_metrics(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    work = frame.copy()
    ranked = work["sales_target"].rank(method="first", pct=True)
    rows = []
    for label, mask in {
        "actual_top_10_pct": ranked > 0.90,
        "actual_top_30_pct": ranked > 0.70,
    }.items():
        group = work.loc[mask]
        y_true = group["sales_target"].to_numpy(dtype=np.float32)
        y_pred = group[prediction_col].to_numpy(dtype=np.float32)
        errors = y_pred - y_true
        row = metric_row(y_true, y_pred, label)
        row["shortage_units"] = float(np.clip(-errors, 0, None).sum())
        row["surplus_units"] = float(np.clip(errors, 0, None).sum())
        row["shortage_units_per_actual_unit"] = row["shortage_units"] / row["actual_sum"] if row["actual_sum"] else math.nan
        row["surplus_units_per_actual_unit"] = row["surplus_units"] / row["actual_sum"] if row["actual_sum"] else math.nan
        rows.append(row)
    capture = tail_capture(frame, prediction_col).iloc[0].to_dict()
    rows.append(
        {
            "slice": "predicted_top_10_pct_capture",
            "rows": capture["rows"],
            "actual_sum": capture["actual_top10_actual_units"],
            "predicted_sum": capture["predicted_top10_actual_units"],
            "unit_ratio_pred_over_actual": capture["actual_top10_unit_recall_by_predicted_top10"],
            "bias_pred_minus_actual": capture["predicted_top10_actual_units"] - capture["actual_top10_actual_units"],
            "actual_mean": math.nan,
            "prediction_mean": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "wape": math.nan,
            "shortage_units": math.nan,
            "surplus_units": math.nan,
            "shortage_units_per_actual_unit": math.nan,
            "surplus_units_per_actual_unit": math.nan,
            "overlap_rate_vs_actual_top10": capture["overlap_rate_vs_actual_top10"],
        }
    )
    return pd.DataFrame.from_records(rows)


def residual_slices(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    rows = []
    group_cols = ["store_chain", "classoftrade", "segment", "subsegment", "title", "store_title_history_bucket"]
    work = frame.copy()
    work["store_title_history_bucket"] = pd.cut(
        work["store_title_prior_obs"],
        bins=[-1, 0, 5, 25, 100, np.inf],
        labels=["cold", "1_5", "6_25", "26_100", "100_plus"],
    ).astype(str)
    for col in group_cols:
        if col not in work.columns:
            continue
        for value, group in work.groupby(col, observed=True, dropna=False):
            if len(group) < 25:
                continue
            row = metric_row(
                group["sales_target"].to_numpy(dtype=np.float32),
                group[prediction_col].to_numpy(dtype=np.float32),
                f"{col}={value}",
            )
            row["group_col"] = col
            row["group_value"] = str(value)
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def write_diagnostics(frame: pd.DataFrame, prediction_col: str, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_decile_metrics(frame, prediction_col).to_csv(output_dir / f"{prefix}_actual_decile_metrics.csv", index=False)
    issue_date_bias(frame, prediction_col).to_csv(output_dir / f"{prefix}_issue_date_bias.csv", index=False)
    calibration_by_prediction_bin(frame, prediction_col).to_csv(output_dir / f"{prefix}_calibration_by_prediction_bin.csv", index=False)
    tail_capture(frame, prediction_col).to_csv(output_dir / f"{prefix}_tail_capture.csv", index=False)
    actual_tail_priority_metrics(frame, prediction_col).to_csv(output_dir / f"{prefix}_tail_priority_metrics.csv", index=False)
    residual_slices(frame, prediction_col).to_csv(output_dir / f"{prefix}_residual_slices.csv", index=False)


def tail_priority_summary(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    metrics = actual_tail_priority_metrics(frame, prediction_col)
    return {str(row["slice"]): row.dropna().to_dict() for _, row in metrics.iterrows()}


def split_by_dates(frame: pd.DataFrame, train_dates: list[str], valid_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_values = frame["onsaledate"].dt.strftime("%Y-%m-%d")
    train_df = frame.loc[date_values.isin(train_dates)].copy()
    valid_df = frame.loc[date_values.isin(valid_dates)].copy()
    return train_df, valid_df


def run_fold_training(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    contract: dict[str, object],
    objective: str,
    rounds: int,
    early_stopping: int,
    seed: int,
    weighting: str,
    output_dir: Path,
    curve_config: CurveObjectiveConfig,
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train_df, valid_df = split_by_dates(frame, fold["train_dates"], fold["validation_dates"])
        artifact = fit_encoder(train_df, contract["numeric_features"], contract["categorical_features"])
        booster, prediction = train_booster(
            train_df,
            valid_df,
            artifact,
            objective,
            rounds,
            early_stopping,
            seed,
            weighting,
            label=f"fold_{fold['fold']}",
            curve_config=curve_config,
        )
        valid_df["prediction"] = prediction
        fold_dir = output_dir / f"fold_{fold['fold']}"
        write_diagnostics(valid_df, "prediction", fold_dir, "validation")
        row = metric_row(
            valid_df["sales_target"].to_numpy(dtype=np.float32),
            valid_df["prediction"].to_numpy(dtype=np.float32),
            "validation",
        )
        row.update(
            {
                "fold": fold["fold"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "best_iteration": int(booster.best_iteration),
                "best_score": float(booster.best_score),
            }
        )
        tail_summary = tail_priority_summary(valid_df, "prediction")
        for key, values in tail_summary.items():
            if key in {"actual_top_10_pct", "actual_top_30_pct"}:
                row[f"{key}_wape"] = values.get("wape")
                row[f"{key}_unit_ratio"] = values.get("unit_ratio_pred_over_actual")
                row[f"{key}_shortage_units"] = values.get("shortage_units")
                row[f"{key}_surplus_units"] = values.get("surplus_units")
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    result.to_csv(output_dir / "fold_metrics.csv", index=False)
    return result


def final_train(
    frame: pd.DataFrame,
    contract: dict[str, object],
    objective: str,
    rounds: int,
    early_stopping: int,
    seed: int,
    weighting: str,
    output_dir: Path,
    curve_config: CurveObjectiveConfig,
) -> dict[str, object]:
    train_df = frame.loc[frame["split"] == "train"].copy()
    valid_df = frame.loc[frame["split"] == "valid"].copy()
    test_df = frame.loc[frame["split"] == "test"].copy()
    artifact = fit_encoder(train_df, contract["numeric_features"], contract["categorical_features"])
    booster, valid_prediction = train_booster(
        train_df,
        valid_df,
        artifact,
        objective,
        rounds,
        early_stopping,
        seed,
        weighting,
        label="final",
        curve_config=curve_config,
    )
    valid_df["prediction"] = valid_prediction
    test_df["prediction"] = predict_frame(booster, test_df, artifact, objective)
    write_diagnostics(valid_df, "prediction", output_dir / "final_validation", "validation")
    write_diagnostics(test_df, "prediction", output_dir / "final_test", "test")
    test_decile_unit_table = log_decile_unit_table(test_df, "prediction", "Final test")
    test_decile_unit_table.to_csv(output_dir / "final_test" / "test_decile_unit_table.csv")
    booster.save_model(output_dir / "model.json")
    (output_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    test_df[
        [
            "store_id",
            "product_id",
            "onsaledate",
            "sales_target",
            "prediction",
            "store_chain",
            "classoftrade",
            "segment",
            "subsegment",
            "title",
        ]
    ].to_csv(output_dir / "test_predictions.csv", index=False)
    return {
        "best_iteration": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "train_weight_report": weight_report(train_df, weighting),
        "validation_weight_report": weight_report(valid_df, weighting),
        "validation": metric_row(valid_df["sales_target"].to_numpy(dtype=np.float32), valid_df["prediction"].to_numpy(dtype=np.float32), "validation"),
        "test": metric_row(test_df["sales_target"].to_numpy(dtype=np.float32), test_df["prediction"].to_numpy(dtype=np.float32), "test"),
        "validation_tail_priority": tail_priority_summary(valid_df, "prediction"),
        "test_tail_priority": tail_priority_summary(test_df, "prediction"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one positive-sales regressor stage.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["regressor_positive"], default="regressor_positive")
    parser.add_argument("--objective", choices=OBJECTIVE_CHOICES, default="log1p_squarederror")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighting", choices=sorted(WEIGHTING_SCHEMES), default="uniform")
    parser.add_argument("--curve-penalty", type=float, default=DEFAULT_CURVE_PENALTY)
    parser.add_argument("--curve-bottom-weight", type=float, default=DEFAULT_CURVE_BOTTOM_WEIGHT)
    parser.add_argument("--curve-middle-weight", type=float, default=DEFAULT_CURVE_MIDDLE_WEIGHT)
    parser.add_argument("--curve-top-weight", type=float, default=DEFAULT_CURVE_TOP_WEIGHT)
    parser.add_argument("--asym-bottom-over-weight", type=float, default=DEFAULT_ASYM_BOTTOM_OVER_WEIGHT)
    parser.add_argument("--asym-bottom-under-weight", type=float, default=DEFAULT_ASYM_BOTTOM_UNDER_WEIGHT)
    parser.add_argument("--asym-middle-weight", type=float, default=DEFAULT_ASYM_MIDDLE_WEIGHT)
    parser.add_argument("--asym-top-under-weight", type=float, default=DEFAULT_ASYM_TOP_UNDER_WEIGHT)
    parser.add_argument("--asym-top-over-weight", type=float, default=DEFAULT_ASYM_TOP_OVER_WEIGHT)
    parser.add_argument("--skip-folds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    curve_config = CurveObjectiveConfig(
        penalty=args.curve_penalty,
        bottom_weight=args.curve_bottom_weight,
        middle_weight=args.curve_middle_weight,
        top_weight=args.curve_top_weight,
        asym_bottom_over_weight=args.asym_bottom_over_weight,
        asym_bottom_under_weight=args.asym_bottom_under_weight,
        asym_middle_weight=args.asym_middle_weight,
        asym_top_under_weight=args.asym_top_under_weight,
        asym_top_over_weight=args.asym_top_over_weight,
    )
    frame, manifest, contract = load_dataset(args.dataset_dir, args.family, args.stage)
    contract, drop_report = apply_feature_drops(contract, args.family)
    run_dir = args.output_dir / args.family.lower() / args.stage / args.objective / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Loaded {len(frame)} rows for {args.family} {args.stage}")
    log(
        "Using "
        f"{drop_report['numeric_features_after']} numeric and {drop_report['categorical_features_after']} categorical features "
        f"after {len(drop_report['active_dropped_numeric_features']) + len(drop_report['active_dropped_categorical_features'])} active feature drops "
        f"and {len(drop_report['configured_non_contract_columns'])} configured non-contract exclusions"
    )
    (run_dir / "feature_drop_report.json").write_text(json.dumps(drop_report, indent=2), encoding="utf-8")
    (run_dir / "feature_contract_effective.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    fold_metrics = pd.DataFrame()
    if not args.skip_folds:
        fold_metrics = run_fold_training(
            frame,
            manifest["folds"],
            contract,
            args.objective,
            args.rounds,
            args.early_stopping,
            args.seed,
            args.weighting,
            run_dir / "folds",
            curve_config,
        )
    final_metrics = final_train(
        frame,
        contract,
        args.objective,
        args.rounds,
        args.early_stopping,
        args.seed,
        args.weighting,
        run_dir,
        curve_config,
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "objective": args.objective,
        "weighting": args.weighting,
        "curve_objective_config": curve_config.to_dict() if args.objective in CURVE_OBJECTIVES else None,
        "dataset_rows": int(len(frame)),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "fold_metrics": fold_metrics.to_dict(orient="records") if not fold_metrics.empty else [],
        "final_metrics": final_metrics,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote model run outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
