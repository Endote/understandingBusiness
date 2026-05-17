#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from train_stage_model import (
    DEFAULT_DATASET_DIR,
    EncodingArtifact,
    best_iteration_end,
    metric_row,
    predict_frame,
    transform_frame,
)
from train_tail_layer import artifact_from_payload, load_base_regressor, probability_rank


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "pipeline_runs"
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]
META_COLUMNS = [
    *KEY_COLUMNS,
    "split",
    "sales_target",
    "positive_sale_flag",
    "store_chain",
    "classoftrade",
    "segment",
    "subsegment",
    "title",
]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_encoding(path: Path) -> EncodingArtifact:
    return artifact_from_payload(load_json(path))


def read_model_frame(dataset_dir: Path, family: str, columns: list[str]) -> pd.DataFrame:
    path = dataset_dir / family.lower() / "classifier_all.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing all-row dataset: {path}")
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    frame["onsaledate"] = pd.to_datetime(frame["onsaledate"])
    return frame


def read_incidence_predictions(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing incidence prediction file: {path}")
    columns = [*KEY_COLUMNS, "raw_probability", "calibrated_probability"]
    frame = pd.read_csv(path, usecols=columns, parse_dates=["onsaledate"])
    duplicates = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        raise ValueError(f"{path} has {duplicates} duplicate key rows for split {split}")
    return frame


def load_tail_model(tail_run_dir: Path) -> tuple[xgb.Booster, EncodingArtifact, dict[str, object]]:
    booster = xgb.Booster()
    booster.load_model(tail_run_dir / "tail_classifier.json")
    artifact = load_encoding(tail_run_dir / "tail_encoding_artifact.json")
    summary = load_json(tail_run_dir / "run_summary.json")
    return booster, artifact, summary


def load_optional_band_model(tail_run_dir: Path) -> xgb.Booster | None:
    model_path = tail_run_dir / "band_classifier.json"
    if not model_path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def load_optional_low_model(tail_run_dir: Path) -> xgb.Booster | None:
    model_path = tail_run_dir / "low_classifier.json"
    if not model_path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def predict_tail_probability(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    return booster.predict(dmatrix, iteration_range=(0, best_iteration_end(booster))).astype(np.float32)


def probability_signal(probability: np.ndarray, mode: str) -> np.ndarray:
    return probability_rank(probability) if mode == "rank" else probability


def apply_tail_multiplier(
    base_prediction: np.ndarray,
    tail_probability: np.ndarray,
    summary: dict[str, object],
    band_probability: np.ndarray | None = None,
    low_probability: np.ndarray | None = None,
    external_signal: np.ndarray | None = None,
    multiplier_override: dict[str, float] | None = None,
) -> np.ndarray:
    best = dict(summary["best_multiplier"])
    if multiplier_override:
        best.update({key: value for key, value in multiplier_override.items() if value is not None})
    alpha = float(best["alpha"])
    band_alpha = float(best.get("band_alpha", 0.0))
    external_alpha = float(best.get("external_alpha", 0.0))
    low_alpha = float(best.get("low_alpha", 0.0))
    scale = float(best["scale"])
    mode = str(summary.get("multiplier_mode", "rank"))
    signal = probability_signal(tail_probability, mode).astype(np.float64)
    log_multiplier = math.log(scale) + alpha * (signal - 0.5)
    if band_probability is not None and band_alpha != 0.0:
        band_signal = probability_signal(band_probability, mode).astype(np.float64)
        log_multiplier = log_multiplier + band_alpha * (band_signal - 0.5)
    if external_signal is not None and external_alpha != 0.0:
        external_rank_signal = probability_signal(external_signal, mode).astype(np.float64)
        log_multiplier = log_multiplier + external_alpha * (external_rank_signal - 0.5)
    if low_probability is not None and low_alpha != 0.0:
        low_signal = probability_signal(low_probability, mode).astype(np.float64)
        log_multiplier = log_multiplier - low_alpha * (low_signal - 0.5)
    multiplier = np.exp(log_multiplier)
    return np.clip(base_prediction.astype(np.float64) * multiplier, 0.0, None).astype(np.float32)


def tail_multiplier_from_summary(
    summary: dict[str, object],
    multiplier_override: dict[str, float] | None = None,
) -> dict[str, float]:
    best = dict(summary["best_multiplier"])
    if multiplier_override:
        best.update({key: value for key, value in multiplier_override.items() if value is not None})
    return {
        "alpha": float(best["alpha"]),
        "band_alpha": float(best.get("band_alpha", 0.0)),
        "external_alpha": float(best.get("external_alpha", 0.0)),
        "low_alpha": float(best.get("low_alpha", 0.0)),
        "scale": float(best["scale"]),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def actual_positive_decile_metrics(frame: pd.DataFrame, prediction_col: str, variant: str, split: str) -> pd.DataFrame:
    positives = frame.loc[frame["sales_target"] > 0].copy()
    if positives.empty:
        return pd.DataFrame()
    positives["actual_positive_decile"] = pd.qcut(
        positives["sales_target"].rank(method="first"),
        q=min(10, len(positives)),
        labels=False,
        duplicates="drop",
    )
    rows: list[dict[str, object]] = []
    for decile, group in positives.groupby("actual_positive_decile", observed=True):
        y_true = group["sales_target"].to_numpy(dtype=np.float32)
        y_pred = group[prediction_col].to_numpy(dtype=np.float32)
        row = metric_row(y_true, y_pred, f"{split}_positive_decile_{int(decile) + 1}")
        row.update(
            {
                "split": split,
                "variant": variant,
                "decile": f"D{int(decile) + 1}",
                "actual_positive_decile": int(decile) + 1,
                "zero_rows": 0,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def all_row_value_decile_metrics(frame: pd.DataFrame, prediction_col: str, variant: str, split: str) -> pd.DataFrame:
    work = frame.copy()
    work["actual_value_decile"] = pd.qcut(
        work["sales_target"].rank(method="first"),
        q=min(10, len(work)),
        labels=False,
        duplicates="drop",
    )
    rows: list[dict[str, object]] = []
    for decile, group in work.groupby("actual_value_decile", observed=True):
        y_true = group["sales_target"].to_numpy(dtype=np.float32)
        y_pred = group[prediction_col].to_numpy(dtype=np.float32)
        row = metric_row(y_true, y_pred, f"{split}_all_row_value_decile_{int(decile) + 1}")
        row.update(
            {
                "split": split,
                "variant": variant,
                "decile": f"D{int(decile) + 1}",
                "actual_value_decile": int(decile) + 1,
                "zero_rows": int((group["sales_target"] <= 0).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def pipeline_score(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    y_true = frame["sales_target"].to_numpy(dtype=np.float32)
    y_pred = frame[prediction_col].to_numpy(dtype=np.float32)
    base = metric_row(y_true, y_pred, "all_rows")
    zero_mask = frame["sales_target"].to_numpy(dtype=np.float32) <= 0
    positive_deciles = actual_positive_decile_metrics(frame, prediction_col, "__tmp__", "__tmp__").sort_values("actual_positive_decile")
    ratios = positive_deciles["unit_ratio_pred_over_actual"].to_numpy(dtype=np.float64)
    bottom = ratios[: min(4, len(ratios))]
    d1_d3 = ratios[: min(3, len(ratios))]
    d4_d6 = ratios[3:6] if len(ratios) >= 6 else np.array([], dtype=np.float64)
    d7_d8 = ratios[6:8] if len(ratios) >= 8 else np.array([], dtype=np.float64)
    top = ratios[6:] if len(ratios) > 6 else np.array([], dtype=np.float64)
    bottom_over = float(np.maximum(bottom - 1.0, 0.0).mean()) if len(bottom) else math.nan
    bottom_under_below_65 = float(np.maximum(0.65 - bottom, 0.0).mean()) if len(bottom) else math.nan
    d7_d8_under = float(np.maximum(1.0 - d7_d8, 0.0).mean()) if len(d7_d8) else math.nan
    top_under = float(np.maximum(1.0 - top, 0.0).mean()) if len(top) else math.nan
    top_over_120 = float(np.maximum(ratios[8:10] - 1.20, 0.0).mean()) if len(ratios) >= 10 else math.nan
    d1_d3_over_115 = float(np.maximum(d1_d3 - 1.15, 0.0).mean()) if len(d1_d3) else math.nan
    d4_d6_below_090 = float(np.maximum(0.90 - d4_d6, 0.0).mean()) if len(d4_d6) else math.nan
    d4_d6_above_110 = float(np.maximum(d4_d6 - 1.10, 0.0).mean()) if len(d4_d6) else math.nan
    d7_d10_below_110 = float(np.maximum(1.10 - top, 0.0).mean()) if len(top) else math.nan
    d7_d10_above_130 = float(np.maximum(top - 1.30, 0.0).mean()) if len(top) else math.nan
    zero_predicted_sum = float(y_pred[zero_mask].sum())
    actual_sum = float(y_true.sum())
    zero_leak_ratio = safe_ratio(zero_predicted_sum, actual_sum)
    total_ratio = float(base["unit_ratio_pred_over_actual"])
    total_gap = abs(total_ratio - 1.0)
    total_below_115 = max(1.15 - total_ratio, 0.0)
    total_above_135 = max(total_ratio - 1.35, 0.0)
    score = (
        7.5 * (d1_d3_over_115 if not math.isnan(d1_d3_over_115) else 0.0)
        + 2.5 * (d4_d6_below_090 if not math.isnan(d4_d6_below_090) else 0.0)
        + 2.5 * (d4_d6_above_110 if not math.isnan(d4_d6_above_110) else 0.0)
        + 7.0 * (d7_d10_below_110 if not math.isnan(d7_d10_below_110) else 0.0)
        + 4.0 * (d7_d10_above_130 if not math.isnan(d7_d10_above_130) else 0.0)
        + 2.5 * total_below_115
        + 2.0 * total_above_135
        + 2.0 * zero_leak_ratio
        + 0.35 * float(base["wape"])
    )
    return {
        "score": float(score),
        "wape": float(base["wape"]),
        "total_unit_ratio": total_ratio,
        "total_unit_ratio_gap": float(total_gap),
        "target_total_below_115": float(total_below_115),
        "target_total_above_135": float(total_above_135),
        "zero_predicted_sum": zero_predicted_sum,
        "zero_leak_ratio": float(zero_leak_ratio),
        "bottom_positive_d1_d4_overprediction": bottom_over,
        "bottom_positive_d1_d4_under_below_65": bottom_under_below_65,
        "positive_d7_d8_underprediction": d7_d8_under,
        "top_positive_d7_d10_underprediction": top_under,
        "positive_d9_d10_overprediction_above_120": top_over_120,
        "target_d1_d3_over_115": d1_d3_over_115,
        "target_d4_d6_below_090": d4_d6_below_090,
        "target_d4_d6_above_110": d4_d6_above_110,
        "target_d7_d10_below_110": d7_d10_below_110,
        "target_d7_d10_above_130": d7_d10_above_130,
        "actual_sum": float(base["actual_sum"]),
        "predicted_sum": float(base["predicted_sum"]),
        "formula": "7.5*d1_d3_over_115 + 2.5*d4_d6_outside_0.90_1.10 + 7*d7_d10_below_1.10 + 4*d7_d10_above_1.30 + 2.5*total_below_1.15 + 2*total_above_1.35 + 2*zero_leak + 0.35*WAPE",
    }


def add_variant(
    variants: dict[str, np.ndarray],
    name: str,
    base: np.ndarray,
    multiplier: np.ndarray | float,
) -> None:
    variants[name] = np.clip(base.astype(np.float64) * multiplier, 0.0, None).astype(np.float32)


def gamma_scale(probability: np.ndarray, gamma: float, target_mean: float) -> float:
    powered = np.power(np.clip(probability.astype(np.float64), 1e-6, 1.0), gamma)
    return float(target_mean / max(float(powered.mean()), 1e-6))


def apply_gamma_scale(probability: np.ndarray, gamma: float, scale: float) -> np.ndarray:
    powered = np.power(np.clip(probability.astype(np.float64), 1e-6, 1.0), gamma)
    return np.clip(powered * scale, 0.0, 1.0).astype(np.float32)


def amount_rank_bucket_multiplier(
    base_prediction: np.ndarray,
    low_factor: float,
    mid_factor: float,
    top_factor: float,
    global_scale: float,
    low_cut: float = 0.45,
    top_cut: float = 0.85,
) -> np.ndarray:
    rank = probability_rank(base_prediction).astype(np.float64)
    multiplier = np.full(len(rank), mid_factor, dtype=np.float64)
    multiplier[rank <= low_cut] = low_factor
    multiplier[rank > top_cut] = top_factor
    return np.clip(global_scale * multiplier, 0.0, 2.0).astype(np.float32)


def add_constrained_soft_gate_amount_variants(
    variants: dict[str, np.ndarray],
    base_prediction: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
) -> None:
    incidence_layers: list[tuple[str, np.ndarray]] = []
    for probability_name, probability in (("raw", raw_probability), ("calibrated", calibrated_probability)):
        for threshold, floor in (
            (0.55, 0.00),
            (0.60, 0.00),
            (0.60, 0.05),
            (0.65, 0.00),
            (0.65, 0.05),
            (0.65, 0.10),
            (0.70, 0.00),
            (0.70, 0.05),
            (0.70, 0.10),
            (0.75, 0.05),
        ):
            gate = (probability >= threshold).astype(np.float32)
            layer = floor + (1.0 - floor) * gate
            incidence_layers.append((f"{probability_name}_t{threshold:.2f}_floor{floor:.2f}", layer.astype(np.float32)))

    amount_profiles = [
        # low positive rows were the oracle failure; D7-D8 need protection; D9-D10 cannot be inflated freely.
        (0.65, 1.05, 0.85, 0.85),
        (0.65, 1.10, 0.85, 0.85),
        (0.70, 1.10, 0.85, 0.90),
        (0.70, 1.15, 0.85, 0.90),
        (0.75, 1.15, 0.90, 0.90),
        (0.75, 1.20, 0.90, 0.90),
        (0.80, 1.10, 0.90, 0.95),
        (0.80, 1.15, 0.90, 0.95),
        (0.85, 1.10, 0.95, 0.95),
        (0.70, 1.00, 0.90, 0.90),
    ]
    for layer_name, incidence_multiplier in incidence_layers:
        for low_factor, mid_factor, top_factor, global_scale in amount_profiles:
            amount_multiplier = amount_rank_bucket_multiplier(
                base_prediction,
                low_factor=low_factor,
                mid_factor=mid_factor,
                top_factor=top_factor,
                global_scale=global_scale,
            )
            name = (
                "G_soft_gate_amount_"
                f"{layer_name}_low{low_factor:.2f}_mid{mid_factor:.2f}_top{top_factor:.2f}_scale{global_scale:.2f}"
            )
            add_variant(variants, name, base_prediction, incidence_multiplier * amount_multiplier)


def add_band_shape_variants(
    variants: dict[str, np.ndarray],
    base_prediction: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    tail_probability: np.ndarray,
    band_probability: np.ndarray | None,
) -> None:
    if band_probability is None:
        return

    amount_rank = probability_rank(base_prediction).astype(np.float64)
    tail_rank = probability_rank(tail_probability).astype(np.float64)
    band_rank = probability_rank(band_probability).astype(np.float64)

    incidence_layers: list[tuple[str, np.ndarray]] = []
    for probability_name, probability in (("raw", raw_probability), ("calibrated", calibrated_probability)):
        for threshold, floor in (
            (0.60, 0.00),
            (0.60, 0.05),
            (0.65, 0.00),
            (0.65, 0.05),
            (0.65, 0.10),
            (0.70, 0.00),
            (0.70, 0.05),
        ):
            gate = (probability >= threshold).astype(np.float64)
            layer = floor + (1.0 - floor) * gate
            incidence_layers.append((f"{probability_name}_t{threshold:.2f}_floor{floor:.2f}", layer.astype(np.float32)))

    profiles = [
        # low_cut, low_factor, lift_rank_min, lift_rank_max, band_cut, lift_factor, tail_cap_cut, cap_rank_min, cap_factor, scale
        (0.34, 0.72, 0.48, 0.86, 0.55, 1.18, 0.88, 0.86, 0.90, 0.95),
        (0.34, 0.78, 0.48, 0.86, 0.55, 1.24, 0.88, 0.86, 0.90, 0.95),
        (0.38, 0.70, 0.50, 0.86, 0.60, 1.25, 0.88, 0.86, 0.88, 0.95),
        (0.38, 0.76, 0.50, 0.88, 0.60, 1.32, 0.90, 0.88, 0.90, 0.95),
        (0.42, 0.68, 0.52, 0.88, 0.62, 1.35, 0.90, 0.88, 0.86, 0.92),
        (0.42, 0.74, 0.52, 0.90, 0.65, 1.42, 0.92, 0.90, 0.88, 0.92),
        (0.46, 0.66, 0.54, 0.90, 0.65, 1.45, 0.92, 0.90, 0.85, 0.90),
        (0.46, 0.72, 0.54, 0.90, 0.70, 1.55, 0.92, 0.90, 0.86, 0.90),
        (0.50, 0.64, 0.56, 0.90, 0.70, 1.60, 0.94, 0.90, 0.84, 0.88),
        (0.50, 0.70, 0.56, 0.92, 0.75, 1.70, 0.94, 0.92, 0.85, 0.88),
    ]

    for layer_name, incidence_multiplier in incidence_layers:
        for (
            low_cut,
            low_factor,
            lift_rank_min,
            lift_rank_max,
            band_cut,
            lift_factor,
            tail_cap_cut,
            cap_rank_min,
            cap_factor,
            scale,
        ) in profiles:
            shape = np.full(len(base_prediction), scale, dtype=np.float64)
            low_mask = amount_rank <= low_cut
            lift_mask = (
                (amount_rank >= lift_rank_min)
                & (amount_rank <= lift_rank_max)
                & (band_rank >= band_cut)
                & (tail_rank < tail_cap_cut)
            )
            cap_mask = (amount_rank >= cap_rank_min) & (tail_rank >= tail_cap_cut)
            shape[low_mask] *= low_factor
            shape[lift_mask] *= lift_factor
            shape[cap_mask] *= cap_factor
            name = (
                "H_band_shape_"
                f"{layer_name}_lowcut{low_cut:.2f}_low{low_factor:.2f}_"
                f"lift{lift_factor:.2f}_band{band_cut:.2f}_cap{cap_factor:.2f}_scale{scale:.2f}"
            )
            add_variant(variants, name, base_prediction, incidence_multiplier * np.clip(shape, 0.0, 2.0))


def build_variants(
    base_prediction: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    tail_probability: np.ndarray | None = None,
    band_probability: np.ndarray | None = None,
    raw_gamma_scales: dict[float, float] | None = None,
) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {"A_regressor_only": base_prediction.astype(np.float32)}
    add_variant(variants, "B_raw_probability_times_regressor", base_prediction, raw_probability)
    add_variant(variants, "C_calibrated_probability_times_regressor", base_prediction, calibrated_probability)

    for gamma in (0.50, 0.65, 0.75, 0.90, 1.10, 1.25, 1.50):
        add_variant(variants, f"D_raw_probability_gamma_{gamma:.2f}", base_prediction, np.power(np.clip(raw_probability, 1e-6, 1.0), gamma))
        add_variant(
            variants,
            f"D_calibrated_probability_gamma_{gamma:.2f}",
            base_prediction,
            np.power(np.clip(calibrated_probability, 1e-6, 1.0), gamma),
        )
        if raw_gamma_scales is not None and gamma in raw_gamma_scales:
            add_variant(
                variants,
                f"D_raw_probability_gamma_{gamma:.2f}_scaled_to_1p20_incidence",
                base_prediction,
                apply_gamma_scale(raw_probability, gamma, raw_gamma_scales[gamma]),
            )

    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80):
        add_variant(variants, f"E_gate_raw_threshold_{threshold:.2f}", base_prediction, (raw_probability >= threshold).astype(np.float32))
        add_variant(
            variants,
            f"E_gate_calibrated_threshold_{threshold:.2f}",
            base_prediction,
            (calibrated_probability >= threshold).astype(np.float32),
        )

    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        raw_gate = (raw_probability >= threshold).astype(np.float32)
        calibrated_gate = (calibrated_probability >= threshold).astype(np.float32)
        for floor in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            add_variant(
                variants,
                f"E_soft_gate_raw_threshold_{threshold:.2f}_floor_{floor:.2f}",
                base_prediction,
                floor + (1.0 - floor) * raw_gate,
            )
            add_variant(
                variants,
                f"E_soft_gate_calibrated_threshold_{threshold:.2f}_floor_{floor:.2f}",
                base_prediction,
                floor + (1.0 - floor) * calibrated_gate,
            )

    for floor in (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        add_variant(variants, f"F_blend_raw_floor_{floor:.2f}", base_prediction, floor + (1.0 - floor) * raw_probability)
        add_variant(
            variants,
            f"F_blend_calibrated_floor_{floor:.2f}",
            base_prediction,
            floor + (1.0 - floor) * calibrated_probability,
        )

    for a in (0.10, 0.20, 0.30, 0.40, 0.50):
        for b in (0.50, 0.75, 1.00, 1.25):
            add_variant(variants, f"F_linear_raw_a_{a:.2f}_b_{b:.2f}", base_prediction, np.clip(a + b * raw_probability, 0.0, 1.5))
            add_variant(
                variants,
                f"F_linear_calibrated_a_{a:.2f}_b_{b:.2f}",
                base_prediction,
                np.clip(a + b * calibrated_probability, 0.0, 1.5),
            )

    add_constrained_soft_gate_amount_variants(variants, base_prediction, raw_probability, calibrated_probability)
    if tail_probability is not None:
        add_band_shape_variants(
            variants,
            base_prediction,
            raw_probability,
            calibrated_probability,
            tail_probability,
            band_probability,
        )

    return variants


def score_variants(frame: pd.DataFrame, variants: dict[str, np.ndarray], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    positive_decile_rows = []
    all_decile_rows = []
    work = frame[META_COLUMNS].copy()
    for name, prediction in variants.items():
        col = "__prediction"
        work[col] = prediction
        summary = pipeline_score(work, col)
        summary.update({"split": split, "variant": name})
        summary_rows.append(summary)
        positive_decile_rows.append(actual_positive_decile_metrics(work, col, name, split))
        all_decile_rows.append(all_row_value_decile_metrics(work, col, name, split))
    return (
        pd.DataFrame.from_records(summary_rows).sort_values("score"),
        pd.concat(positive_decile_rows, ignore_index=True),
        pd.concat(all_decile_rows, ignore_index=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all-row incidence classifier x positive regressor pipeline variants.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--tail-run-dir", type=Path, required=True)
    parser.add_argument("--calibration-run-dir", type=Path, required=True)
    parser.add_argument("--external-signal-run-dir", type=Path, default=None)
    parser.add_argument("--tail-alpha", type=float, default=None)
    parser.add_argument("--tail-band-alpha", type=float, default=None)
    parser.add_argument("--tail-external-alpha", type=float, default=None)
    parser.add_argument("--tail-low-alpha", type=float, default=None)
    parser.add_argument("--tail-scale", type=float, default=None)
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_booster, base_artifact, base_objective = load_base_regressor(args.base_run_dir)
    tail_booster, tail_artifact, tail_summary = load_tail_model(args.tail_run_dir)
    band_booster = load_optional_band_model(args.tail_run_dir)
    low_booster = load_optional_low_model(args.tail_run_dir)
    external_signal_run_dir = args.external_signal_run_dir
    if external_signal_run_dir is None and tail_summary.get("external_signal_run_dir"):
        external_signal_run_dir = Path(str(tail_summary["external_signal_run_dir"]))
    external_booster = None
    external_artifact = None
    external_objective = None
    if external_signal_run_dir is not None:
        external_booster, external_artifact, external_objective = load_base_regressor(external_signal_run_dir)
    multiplier_override = {
        "alpha": args.tail_alpha,
        "band_alpha": args.tail_band_alpha,
        "external_alpha": args.tail_external_alpha,
        "low_alpha": args.tail_low_alpha,
        "scale": args.tail_scale,
    }

    needed_columns = [
        *META_COLUMNS,
        *base_artifact.numeric_features,
        *base_artifact.categorical_features,
        *tail_artifact.numeric_features,
        *tail_artifact.categorical_features,
    ]
    if external_artifact is not None:
        needed_columns.extend([*external_artifact.numeric_features, *external_artifact.categorical_features])
    frame = read_model_frame(args.dataset_dir, args.family, needed_columns)
    frame = frame.loc[frame["split"].isin(["valid", "test"])].copy()

    valid_incidence = read_incidence_predictions(args.calibration_run_dir / "validation_calibrated_predictions.csv", "valid")
    test_incidence = read_incidence_predictions(args.calibration_run_dir / "test_calibrated_predictions.csv", "test")
    incidence = pd.concat([valid_incidence, test_incidence], ignore_index=True)
    frame = frame.merge(incidence, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if frame[["raw_probability", "calibrated_probability"]].isna().any().any():
        missing = int(frame["raw_probability"].isna().sum())
        raise ValueError(f"Missing incidence probabilities after merge: {missing} rows")

    run_dir = args.output_dir / args.family.lower() / "incidence_grid" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Scoring all-row pipeline grid into {run_dir}")

    split_outputs = {}
    validation_summary_for_selection = None
    raw_gamma_scales = None
    for split in ("valid", "test"):
        split_df = frame.loc[frame["split"] == split].copy()
        log(f"Predicting positive regressor and tail layer for {split}: {len(split_df)} rows")
        base_prediction = predict_frame(base_booster, split_df, base_artifact, base_objective)
        tail_probability = predict_tail_probability(tail_booster, split_df, tail_artifact)
        band_probability = predict_tail_probability(band_booster, split_df, tail_artifact) if band_booster is not None else None
        low_probability = predict_tail_probability(low_booster, split_df, tail_artifact) if low_booster is not None else None
        external_signal = (
            predict_frame(external_booster, split_df, external_artifact, external_objective)
            if external_booster is not None and external_artifact is not None and external_objective is not None
            else None
        )
        regressor_prediction = apply_tail_multiplier(
            base_prediction,
            tail_probability,
            tail_summary,
            band_probability=band_probability,
            low_probability=low_probability,
            external_signal=external_signal,
            multiplier_override=multiplier_override,
        )
        split_df["positive_regressor_prediction"] = regressor_prediction
        split_df["tail_probability"] = tail_probability
        if band_probability is not None:
            split_df["band_probability"] = band_probability
        if low_probability is not None:
            split_df["low_probability"] = low_probability
        if external_signal is not None:
            split_df["external_signal_prediction"] = external_signal
        if split == "valid":
            valid_target_probability_mean = min(1.20 * float(split_df["positive_sale_flag"].mean()), 1.0)
            raw_gamma_scales = {
                gamma: gamma_scale(split_df["raw_probability"].to_numpy(dtype=np.float32), gamma, valid_target_probability_mean)
                for gamma in (0.50, 0.65, 0.75, 0.90, 1.10, 1.25, 1.50)
            }
        if raw_gamma_scales is None:
            raise RuntimeError("Validation gamma scales were not initialized before test scoring.")
        variants = build_variants(
            regressor_prediction,
            split_df["raw_probability"].to_numpy(dtype=np.float32),
            split_df["calibrated_probability"].to_numpy(dtype=np.float32),
            tail_probability=tail_probability,
            band_probability=band_probability,
            raw_gamma_scales=raw_gamma_scales,
        )
        add_variant(
            variants,
            "ORACLE_actual_positive_gate",
            regressor_prediction,
            split_df["positive_sale_flag"].to_numpy(dtype=np.float32),
        )
        summary, positive_deciles, all_deciles = score_variants(split_df, variants, split)
        summary.to_csv(run_dir / f"{split}_variant_summary.csv", index=False)
        positive_deciles.to_csv(run_dir / f"{split}_positive_actual_decile_metrics.csv", index=False)
        all_deciles.to_csv(run_dir / f"{split}_all_row_actual_value_decile_metrics.csv", index=False)
        split_outputs[split] = {
            "summary": summary,
            "positive_deciles": positive_deciles,
            "all_deciles": all_deciles,
            "frame": split_df,
            "variants": variants,
        }
        if split == "valid":
            validation_summary_for_selection = summary
        if args.write_predictions:
            signal_columns = ["raw_probability", "calibrated_probability", "positive_regressor_prediction", "tail_probability"]
            if band_probability is not None:
                signal_columns.append("band_probability")
            if low_probability is not None:
                signal_columns.append("low_probability")
            pred_frame = split_df[META_COLUMNS + signal_columns].copy()
            for name, prediction in variants.items():
                pred_frame[name] = prediction
            pred_frame.to_csv(run_dir / f"{split}_pipeline_predictions.csv", index=False)

    assert validation_summary_for_selection is not None
    selectable = validation_summary_for_selection.loc[
        ~validation_summary_for_selection["variant"].astype(str).str.startswith("ORACLE_")
    ]
    selected_variant = str(selectable.iloc[0]["variant"])
    selected_validation_summary = selectable.iloc[0].to_dict()
    selected_test_summary = split_outputs["test"]["summary"].loc[
        split_outputs["test"]["summary"]["variant"] == selected_variant
    ].iloc[0].to_dict()
    log(f"Selected by validation score: {selected_variant}")

    selected_test_deciles = split_outputs["test"]["positive_deciles"].loc[
        split_outputs["test"]["positive_deciles"]["variant"] == selected_variant
    ]
    selected_test_deciles.to_csv(run_dir / "selected_test_positive_actual_decile_metrics.csv", index=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "base_run_dir": str(args.base_run_dir),
        "tail_run_dir": str(args.tail_run_dir),
        "calibration_run_dir": str(args.calibration_run_dir),
        "external_signal_run_dir": str(external_signal_run_dir) if external_signal_run_dir is not None else None,
        "selected_variant_by_validation_score": selected_variant,
        "validation_selected_summary": selected_validation_summary,
        "test_selected_summary": selected_test_summary,
        "base_objective": base_objective,
        "external_signal_objective": external_objective,
        "tail_multiplier": tail_multiplier_from_summary(tail_summary, multiplier_override),
        "tail_multiplier_source": "cli_override" if any(value is not None for value in multiplier_override.values()) else "tail_run_best_multiplier",
        "band_layer_loaded": band_booster is not None,
        "low_layer_loaded": low_booster is not None,
        "external_signal_loaded": external_booster is not None,
        "scoring_formula": "7.5*d1_d3_over_115 + 2.5*d4_d6_outside_0.90_1.10 + 7*d7_d10_below_1.10 + 4*d7_d10_above_1.30 + 2.5*total_below_1.15 + 2*total_above_1.35 + 2*zero_leak + 0.35*WAPE",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
