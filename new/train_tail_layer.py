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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from train_stage_model import (
    DEFAULT_DATASET_DIR,
    EncodingArtifact,
    actual_decile_metrics,
    apply_feature_drops,
    composite_decile_score,
    fit_encoder,
    inverse_predictions,
    load_dataset,
    log_decile_unit_table,
    metric_row,
    predict_frame,
    transform_frame,
    write_diagnostics,
)


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "tail_layer_runs"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def artifact_from_payload(payload: dict[str, object]) -> EncodingArtifact:
    return EncodingArtifact(
        numeric_features=list(payload["numeric_features"]),
        categorical_features=list(payload["categorical_features"]),
        numeric_medians={str(key): float(value) for key, value in dict(payload["numeric_medians"]).items()},
        category_maps={
            str(col): {str(level): int(code) for level, code in dict(mapping).items()}
            for col, mapping in dict(payload["category_maps"]).items()
        },
        category_sizes={str(key): int(value) for key, value in dict(payload["category_sizes"]).items()},
        feature_names=list(payload["feature_names"]),
    )


def load_base_regressor(base_run_dir: Path) -> tuple[xgb.Booster, EncodingArtifact, str]:
    summary = json.loads((base_run_dir / "run_summary.json").read_text(encoding="utf-8"))
    artifact = artifact_from_payload(json.loads((base_run_dir / "encoding_artifact.json").read_text(encoding="utf-8")))
    booster = xgb.Booster()
    booster.load_model(base_run_dir / "model.json")
    return booster, artifact, str(summary["objective"])


def tail_labels(frame: pd.DataFrame, quantile: float) -> np.ndarray:
    ranks = frame["sales_target"].rank(method="first", pct=True).to_numpy(dtype=np.float32)
    return (ranks > quantile).astype(np.float32)


def threshold_tail_labels(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return (frame["sales_target"].to_numpy(dtype=np.float32) >= threshold).astype(np.float32)


def target_tail_labels(frame: pd.DataFrame, quantile: float, threshold: float | None) -> np.ndarray:
    if threshold is not None:
        return threshold_tail_labels(frame, threshold)
    return tail_labels(frame, quantile)


def tail_target_name(quantile: float, threshold: float | None) -> str:
    if threshold is not None:
        threshold_text = str(threshold).replace(".", "p")
        return f"tail_ge{threshold_text}"
    return f"tail_q{int(quantile * 100)}"


def band_labels(frame: pd.DataFrame, lower_quantile: float, upper_quantile: float) -> np.ndarray:
    ranks = frame["sales_target"].rank(method="first", pct=True).to_numpy(dtype=np.float32)
    return ((ranks > lower_quantile) & (ranks <= upper_quantile)).astype(np.float32)


def threshold_band_labels(frame: pd.DataFrame, minimum: float, maximum: float | None) -> np.ndarray:
    target = frame["sales_target"].to_numpy(dtype=np.float32)
    mask = target >= minimum
    if maximum is not None:
        mask &= target <= maximum
    return mask.astype(np.float32)


def low_target_labels(frame: pd.DataFrame, maximum: float) -> np.ndarray:
    return (frame["sales_target"].to_numpy(dtype=np.float32) <= maximum).astype(np.float32)


def low_target_name(maximum: float | None) -> str:
    if maximum is None:
        return "low_disabled"
    max_text = str(maximum).replace(".", "p")
    return f"low_le{max_text}"


def target_band_labels(
    frame: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
    minimum: float | None,
    maximum: float | None,
) -> np.ndarray:
    if minimum is not None:
        return threshold_band_labels(frame, minimum, maximum)
    return band_labels(frame, lower_quantile, upper_quantile)


def band_target_name(lower_quantile: float, upper_quantile: float, minimum: float | None, maximum: float | None) -> str:
    if minimum is None:
        return f"band_q{int(lower_quantile * 100)}_q{int(upper_quantile * 100)}"
    min_text = str(minimum).replace(".", "p")
    if maximum is None:
        return f"band_ge{min_text}"
    max_text = str(maximum).replace(".", "p")
    return f"band_{min_text}_to_{max_text}"


def train_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    artifact: EncodingArtifact,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    label: str,
    rounds: int,
    early_stopping: int,
    seed: int,
    max_depth: int,
    min_child_weight: float,
    eta: float,
    subsample: float,
    colsample_bytree: float,
    scale_pos_weight_multiplier: float,
) -> tuple[xgb.Booster, np.ndarray, np.ndarray]:
    train_encoded = transform_frame(train_df, artifact, "tweedie", "uniform")
    valid_encoded = transform_frame(valid_df, artifact, "tweedie", "uniform")
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = (negatives / positives if positives > 0 else 1.0) * scale_pos_weight_multiplier
    dtrain = xgb.DMatrix(train_encoded.matrix, label=y_train, feature_names=artifact.feature_names)
    dvalid = xgb.DMatrix(valid_encoded.matrix, label=y_valid, feature_names=artifact.feature_names)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist",
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
    }
    log(
        f"Training {label} classifier: "
        f"{len(train_df)} train rows, {len(valid_df)} validation rows, "
        f"positive_rate {positives / max(len(y_train), 1):.4f}, scale_pos_weight {scale_pos_weight:.3f}"
    )
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping,
        verbose_eval=25,
    )
    valid_prob = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    train_prob = booster.predict(dtrain, iteration_range=(0, booster.best_iteration + 1))
    return booster, train_prob, valid_prob


def predict_classifier(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration + 1))


def classifier_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    y_true: np.ndarray,
    classifier_target: str,
    label: str,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "slice": label,
        "rows": int(len(frame)),
        "classifier_target": classifier_target,
        "positive_rate": float(y_true.mean()) if len(y_true) else math.nan,
        "probability_mean": float(np.mean(probability)) if len(probability) else math.nan,
        "probability_p90": float(np.quantile(probability, 0.90)) if len(probability) else math.nan,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        metrics["average_precision"] = float(average_precision_score(y_true, probability))
        metrics["log_loss"] = float(log_loss(y_true, np.clip(probability, 1e-6, 1 - 1e-6)))
    else:
        metrics["roc_auc"] = math.nan
        metrics["average_precision"] = math.nan
        metrics["log_loss"] = math.nan
    for top_frac in (0.10, 0.20, 0.30):
        n = max(1, math.ceil(len(frame) * top_frac))
        idx = np.argsort(probability)[-n:]
        metrics[f"actual_tail_recall_at_pred_top_{int(top_frac * 100)}"] = float(y_true[idx].sum() / max(y_true.sum(), 1.0))
        metrics[f"actual_units_at_pred_top_{int(top_frac * 100)}"] = float(frame.iloc[idx]["sales_target"].sum())
    return metrics


def probability_rank(probability: np.ndarray) -> np.ndarray:
    if len(probability) == 0:
        return probability.astype(np.float32)
    order = np.argsort(probability, kind="stable")
    ranks = np.empty(len(probability), dtype=np.float32)
    ranks[order] = (np.arange(len(probability), dtype=np.float32) + 0.5) / len(probability)
    return ranks


def probability_signal(probability: np.ndarray, mode: str) -> np.ndarray:
    return probability_rank(probability) if mode == "rank" else probability


def adjusted_predictions(base_prediction: np.ndarray, probability: np.ndarray, alpha: float, scale: float, mode: str) -> np.ndarray:
    signal = probability_signal(probability, mode)
    return adjusted_predictions_from_signal(base_prediction, signal, alpha=alpha, scale=scale)


def adjusted_predictions_from_signal(base_prediction: np.ndarray, signal: np.ndarray, alpha: float, scale: float) -> np.ndarray:
    multiplier = scale * np.exp(alpha * (signal - 0.5))
    return np.clip(base_prediction * multiplier, 0.0, None)


def adjusted_predictions_from_signals(
    base_prediction: np.ndarray,
    tail_signal: np.ndarray,
    tail_alpha: float,
    scale: float,
    band_signal: np.ndarray | None = None,
    band_alpha: float = 0.0,
    external_signal: np.ndarray | None = None,
    external_alpha: float = 0.0,
    low_signal: np.ndarray | None = None,
    low_alpha: float = 0.0,
) -> np.ndarray:
    log_multiplier = math.log(scale) + tail_alpha * (tail_signal - 0.5)
    if band_signal is not None and band_alpha != 0.0:
        log_multiplier = log_multiplier + band_alpha * (band_signal - 0.5)
    if external_signal is not None and external_alpha != 0.0:
        log_multiplier = log_multiplier + external_alpha * (external_signal - 0.5)
    if low_signal is not None and low_alpha != 0.0:
        log_multiplier = log_multiplier - low_alpha * (low_signal - 0.5)
    return np.clip(base_prediction * np.exp(log_multiplier), 0.0, None)


def curve_score(frame: pd.DataFrame, prediction_col: str) -> float:
    metrics = actual_decile_metrics(frame, prediction_col).sort_values("actual_decile")
    ratios = metrics["unit_ratio_pred_over_actual"].to_numpy(dtype=np.float64)
    total_ratio = float(frame[prediction_col].sum() / frame["sales_target"].sum())
    score = 0.0
    for idx, ratio in enumerate(ratios):
        decile = idx + 1
        if decile <= 4:
            score += (ratio - 1.0) * 1.7 if ratio > 1.0 else (1.0 - ratio) * 0.5
        elif decile >= 6:
            score += (1.0 - ratio) * 2.2 if ratio < 1.0 else (ratio - 1.0) * 0.6
        else:
            score += abs(ratio - 1.0) * 0.7
    score += abs(total_ratio - 1.0) * 0.6
    return float(score)


def strict_composite_decile_score(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    base = composite_decile_score(frame, prediction_col)
    deciles = actual_decile_metrics(frame, prediction_col).sort_values("actual_decile")
    ratios = deciles["unit_ratio_pred_over_actual"].to_numpy(dtype=np.float64)
    d7_d8 = ratios[6:8] if len(ratios) >= 8 else np.array([], dtype=np.float64)
    d9_d10 = ratios[8:10] if len(ratios) >= 10 else np.array([], dtype=np.float64)
    d7_d8_under = float(np.maximum(1.0 - d7_d8, 0.0).mean()) if len(d7_d8) else math.nan
    d7_d8_worst_under = float(np.maximum(1.0 - d7_d8, 0.0).max()) if len(d7_d8) else math.nan
    d9_d10_over_110 = float(np.maximum(d9_d10 - 1.10, 0.0).mean()) if len(d9_d10) else math.nan
    score = (
        float(base["score"])
        + 4.0 * (d7_d8_under if not math.isnan(d7_d8_under) else 0.0)
        + 2.0 * (d7_d8_worst_under if not math.isnan(d7_d8_worst_under) else 0.0)
        + 1.5 * (d9_d10_over_110 if not math.isnan(d9_d10_over_110) else 0.0)
    )
    return {
        "score": float(score),
        "base_composite_decile_score": float(base["score"]),
        "d7_d8_underprediction": d7_d8_under,
        "d7_d8_worst_underprediction": d7_d8_worst_under,
        "d9_d10_overprediction_above_110": d9_d10_over_110,
        "formula": "composite + 4.0*mean(max(1-D7_D8,0)) + 2.0*max(max(1-D7_D8,0)) + 1.5*mean(max(D9_D10-1.10,0))",
    }


def unit_ratio(actual: pd.Series, predicted: pd.Series) -> float:
    actual_sum = float(actual.sum())
    return float(predicted.sum() / actual_sum) if actual_sum > 0 else math.nan


def sip_band_metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    target = frame["sales_target"]
    prediction = frame[prediction_col]
    rank = target.rank(method="first", pct=True)
    masks = {
        "bottom30": rank <= 0.30,
        "top30": rank > 0.70,
        "top10": rank > 0.90,
        "target_1": target == 1,
        "target_2": target == 2,
        "target_3": target == 3,
        "target_4": target == 4,
        "target_5_plus": target >= 5,
        "target_3_plus": target >= 3,
    }
    metrics: dict[str, object] = {}
    for name, mask in masks.items():
        metrics[f"{name}_rows"] = int(mask.sum())
        metrics[f"{name}_actual_sum"] = float(target.loc[mask].sum())
        metrics[f"{name}_predicted_sum"] = float(prediction.loc[mask].sum())
        metrics[f"{name}_ratio"] = unit_ratio(target.loc[mask], prediction.loc[mask])
    return metrics


def sip_band_score(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    metrics = sip_band_metrics(frame, prediction_col)
    bottom30 = float(metrics["bottom30_ratio"])
    top30 = float(metrics["top30_ratio"])
    top10 = float(metrics["top10_ratio"])
    target_1 = float(metrics["target_1_ratio"])
    target_2 = float(metrics["target_2_ratio"])
    target_3 = float(metrics["target_3_ratio"])
    target_4 = float(metrics["target_4_ratio"])
    target_5_plus = float(metrics["target_5_plus_ratio"])
    target_3_plus = float(metrics["target_3_plus_ratio"])
    score = (
        60.0 * max(bottom30 - 1.30, 0.0)
        + 20.0 * max(0.65 - top30, 0.0)
        + 16.0 * max(0.67 - top10, 0.0)
        + 8.0 * max(target_1 - 1.25, 0.0)
        + 2.0 * max(0.75 - target_2, 0.0)
        + 2.0 * max(target_2 - 1.20, 0.0)
        + 8.0 * max(0.63 - target_3_plus, 0.0)
        + 10.0 * max(0.60 - target_3, 0.0)
        + 4.0 * max(0.60 - target_4, 0.0)
        + 4.0 * max(0.60 - target_5_plus, 0.0)
        + 0.4 * abs(bottom30 - 1.25)
        + 0.8 * abs(top30 - 0.70)
        + 0.6 * abs(top10 - 0.72)
    )
    return {
        "score": float(score),
        "formula": (
            "60*max(bottom30-1.30,0) + 20*max(0.65-top30,0) + "
            "16*max(0.67-top10,0) + band penalties for target=1,2,3,4,5+"
        ),
        **metrics,
    }


def sip_top_consensus_score(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    metrics = sip_band_metrics(frame, prediction_col)
    bottom30 = float(metrics["bottom30_ratio"])
    top30 = float(metrics["top30_ratio"])
    top10 = float(metrics["top10_ratio"])
    target_1 = float(metrics["target_1_ratio"])
    target_2 = float(metrics["target_2_ratio"])
    target_3 = float(metrics["target_3_ratio"])
    target_4 = float(metrics["target_4_ratio"])
    target_5_plus = float(metrics["target_5_plus_ratio"])
    target_3_plus = float(metrics["target_3_plus_ratio"])
    score = (
        80.0 * max(bottom30 - 1.42, 0.0)
        + 28.0 * max(target_1 - 1.35, 0.0)
        + 34.0 * max(0.82 - top30, 0.0)
        + 30.0 * max(0.88 - top10, 0.0)
        + 8.0 * max(0.70 - target_3_plus, 0.0)
        + 5.0 * max(0.68 - target_3, 0.0)
        + 4.0 * max(0.74 - target_4, 0.0)
        + 4.0 * max(0.80 - target_5_plus, 0.0)
        + 2.0 * max(0.85 - target_2, 0.0)
        + 0.5 * abs(bottom30 - 1.34)
        + 0.8 * abs(top30 - 0.86)
        + 0.8 * abs(top10 - 0.92)
    )
    return {
        "score": float(score),
        "formula": (
            "80*max(bottom30-1.42,0) + 28*max(target_1-1.35,0) + "
            "34*max(0.82-top30,0) + 30*max(0.88-top10,0) + top-band penalties"
        ),
        **metrics,
    }


def sip_top_push_score(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    metrics = sip_band_metrics(frame, prediction_col)
    bottom30 = float(metrics["bottom30_ratio"])
    top30 = float(metrics["top30_ratio"])
    top10 = float(metrics["top10_ratio"])
    target_1 = float(metrics["target_1_ratio"])
    target_2 = float(metrics["target_2_ratio"])
    target_3 = float(metrics["target_3_ratio"])
    target_4 = float(metrics["target_4_ratio"])
    target_5_plus = float(metrics["target_5_plus_ratio"])
    target_3_plus = float(metrics["target_3_plus_ratio"])
    score = (
        85.0 * max(bottom30 - 1.50, 0.0)
        + 36.0 * max(target_1 - 1.43, 0.0)
        + 38.0 * max(0.86 - top30, 0.0)
        + 34.0 * max(0.92 - top10, 0.0)
        + 8.0 * max(0.74 - target_3_plus, 0.0)
        + 5.0 * max(0.72 - target_3, 0.0)
        + 4.0 * max(0.80 - target_4, 0.0)
        + 4.0 * max(0.88 - target_5_plus, 0.0)
        + 2.0 * max(0.90 - target_2, 0.0)
        + 0.4 * abs(bottom30 - 1.45)
        + 1.0 * abs(top30 - 0.88)
        + 1.0 * abs(top10 - 0.94)
    )
    return {
        "score": float(score),
        "formula": (
            "85*max(bottom30-1.50,0) + 36*max(target_1-1.43,0) + "
            "38*max(0.86-top30,0) + 34*max(0.92-top10,0) + relaxed top-push penalties"
        ),
        **metrics,
    }


def sip_tail_gain_score(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    metrics = sip_band_metrics(frame, prediction_col)
    bottom30 = float(metrics["bottom30_ratio"])
    top30 = float(metrics["top30_ratio"])
    top10 = float(metrics["top10_ratio"])
    target_1 = float(metrics["target_1_ratio"])
    target_2 = float(metrics["target_2_ratio"])
    target_3 = float(metrics["target_3_ratio"])
    target_4 = float(metrics["target_4_ratio"])
    target_5_plus = float(metrics["target_5_plus_ratio"])
    target_3_plus = float(metrics["target_3_plus_ratio"])
    score = (
        95.0 * max(bottom30 - 1.50, 0.0)
        + 42.0 * max(target_1 - 1.48, 0.0)
        + 34.0 * max(0.88 - top30, 0.0)
        + 30.0 * max(0.94 - top10, 0.0)
        + 8.0 * max(0.76 - target_3_plus, 0.0)
        + 5.0 * max(0.74 - target_3, 0.0)
        + 4.0 * max(0.82 - target_4, 0.0)
        + 4.0 * max(0.92 - target_5_plus, 0.0)
        + 2.0 * max(0.92 - target_2, 0.0)
        + 0.25 * abs(bottom30 - 1.45)
        + 0.7 * abs(top30 - 0.90)
        + 0.7 * abs(top10 - 0.96)
    )
    return {
        "score": float(score),
        "formula": (
            "95*max(bottom30-1.50,0) + 42*max(target_1-1.48,0) + "
            "34*max(0.88-top30,0) + 30*max(0.94-top10,0) + tail-gain penalties"
        ),
        **metrics,
    }


def calibration_grid(
    valid_df: pd.DataFrame,
    base_prediction: np.ndarray,
    tail_probability: np.ndarray,
    band_probability: np.ndarray | None,
    external_probability: np.ndarray | None,
    low_probability: np.ndarray | None,
    mode: str,
    alphas: list[float],
    band_alphas: list[float],
    external_alphas: list[float],
    low_alphas: list[float],
    scales: list[float],
    selection_metric: str,
) -> pd.DataFrame:
    rows = []
    work = valid_df.copy()
    work["base_prediction"] = base_prediction
    tail_signal = probability_signal(tail_probability, mode)
    band_signal = probability_signal(band_probability, mode) if band_probability is not None else None
    external_signal = probability_signal(external_probability, mode) if external_probability is not None else None
    low_signal = probability_signal(low_probability, mode) if low_probability is not None else None
    for alpha in alphas:
        for band_alpha in band_alphas:
            for external_alpha in external_alphas:
                for low_alpha in low_alphas:
                    for scale in scales:
                        col = "adjusted_prediction"
                        work[col] = adjusted_predictions_from_signals(
                            base_prediction,
                            tail_signal,
                            tail_alpha=alpha,
                            scale=scale,
                            band_signal=band_signal,
                            band_alpha=band_alpha,
                            external_signal=external_signal,
                            external_alpha=external_alpha,
                            low_signal=low_signal,
                            low_alpha=low_alpha,
                        )
                        row = metric_row(
                            work["sales_target"].to_numpy(dtype=np.float32),
                            work[col].to_numpy(dtype=np.float32),
                            "validation_adjusted",
                        )
                        deciles = actual_decile_metrics(work, col).sort_values("actual_decile")
                        for _, decile_row in deciles.iterrows():
                            row[f"d{int(decile_row['actual_decile']) + 1}_ratio"] = decile_row["unit_ratio_pred_over_actual"]
                        row["curve_score"] = curve_score(work, col)
                        row["composite_decile_score"] = composite_decile_score(work, col)["score"]
                        row["strict_composite_decile_score"] = strict_composite_decile_score(work, col)["score"]
                        sip_score = sip_band_score(work, col)
                        row["sip_band_score"] = sip_score["score"]
                        consensus_score = sip_top_consensus_score(work, col)
                        row["sip_top_consensus_score"] = consensus_score["score"]
                        top_push_score = sip_top_push_score(work, col)
                        row["sip_top_push_score"] = top_push_score["score"]
                        tail_gain_score = sip_tail_gain_score(work, col)
                        row["sip_tail_gain_score"] = tail_gain_score["score"]
                        for key, value in sip_score.items():
                            if key.endswith("_ratio"):
                                row[f"sip_{key}"] = value
                        row["alpha"] = alpha
                        row["band_alpha"] = band_alpha
                        row["external_alpha"] = external_alpha
                        row["low_alpha"] = low_alpha
                        row["scale"] = scale
                        rows.append(row)
    sort_col = {
        "composite": "composite_decile_score",
        "strict": "strict_composite_decile_score",
        "sip_band": "sip_band_score",
        "sip_top_consensus": "sip_top_consensus_score",
        "sip_top_push": "sip_top_push_score",
        "sip_tail_gain": "sip_tail_gain_score",
    }[selection_metric]
    return pd.DataFrame.from_records(rows).sort_values(sort_col)


def group_key_frame(frame: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["__GLOBAL__"] * len(frame), index=frame.index, dtype="string")
    missing = [col for col in group_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing group calibration columns: {missing}")
    return frame[group_cols].astype("string").fillna("UNKNOWN").agg("||".join, axis=1)


def grouped_calibration_table(
    valid_df: pd.DataFrame,
    base_prediction: np.ndarray,
    tail_probability: np.ndarray,
    mode: str,
    group_cols: list[str],
    alphas: list[float],
    scales: list[float],
    global_alpha: float,
    global_scale: float,
    min_rows: int,
    shrinkage: float,
) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame()

    signal = probability_signal(tail_probability, mode)
    keys = group_key_frame(valid_df, group_cols)
    rows = []
    for key, index in keys.groupby(keys, sort=False).groups.items():
        positions = valid_df.index.get_indexer(index)
        group = valid_df.loc[index].copy()
        group_base = base_prediction[positions]
        group_signal = signal[positions]
        if len(group) < min_rows or float(group["sales_target"].sum()) <= 0:
            continue

        best_row = None
        for alpha in alphas:
            for scale in scales:
                group["adjusted_prediction"] = adjusted_predictions_from_signal(group_base, group_signal, alpha=alpha, scale=scale)
                row = metric_row(
                    group["sales_target"].to_numpy(dtype=np.float32),
                    group["adjusted_prediction"].to_numpy(dtype=np.float32),
                    "validation_group_adjusted",
                )
                row["curve_score"] = curve_score(group, "adjusted_prediction")
                row["composite_decile_score"] = composite_decile_score(group, "adjusted_prediction")["score"]
                row["alpha"] = alpha
                row["scale"] = scale
                if best_row is None or row["composite_decile_score"] < best_row["composite_decile_score"]:
                    best_row = row
        if best_row is None:
            continue

        shrink = float(len(group) / (len(group) + max(shrinkage, 0.0)))
        shrunk_alpha = global_alpha + shrink * (float(best_row["alpha"]) - global_alpha)
        shrunk_scale = float(np.exp(np.log(global_scale) + shrink * (np.log(float(best_row["scale"])) - np.log(global_scale))))
        rows.append(
            {
                "group_key": str(key),
                "rows": int(len(group)),
                "actual_sum": float(group["sales_target"].sum()),
                "raw_alpha": float(best_row["alpha"]),
                "raw_scale": float(best_row["scale"]),
                "raw_curve_score": float(best_row["curve_score"]),
                "raw_composite_decile_score": float(best_row["composite_decile_score"]),
                "shrinkage_weight": shrink,
                "alpha": shrunk_alpha,
                "scale": shrunk_scale,
            }
        )
    return pd.DataFrame.from_records(rows)


def apply_grouped_adjustment(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    tail_probability: np.ndarray,
    band_probability: np.ndarray | None,
    external_probability: np.ndarray | None,
    low_probability: np.ndarray | None,
    mode: str,
    group_cols: list[str],
    group_table: pd.DataFrame,
    global_alpha: float,
    global_scale: float,
    global_band_alpha: float = 0.0,
    global_external_alpha: float = 0.0,
    global_low_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = probability_signal(tail_probability, mode)
    band_signal = probability_signal(band_probability, mode) if band_probability is not None else None
    external_signal = probability_signal(external_probability, mode) if external_probability is not None else None
    low_signal = probability_signal(low_probability, mode) if low_probability is not None else None
    alpha = np.full(len(frame), global_alpha, dtype=np.float64)
    scale = np.full(len(frame), global_scale, dtype=np.float64)
    if group_cols and not group_table.empty:
        params = group_table.set_index("group_key")[["alpha", "scale"]].to_dict(orient="index")
        keys = group_key_frame(frame, group_cols)
        for key, index in keys.groupby(keys, sort=False).groups.items():
            values = params.get(str(key))
            if values is None:
                continue
            positions = frame.index.get_indexer(index)
            alpha[positions] = float(values["alpha"])
            scale[positions] = float(values["scale"])
    multiplier = scale * np.exp(alpha * (signal - 0.5))
    if band_signal is not None and global_band_alpha != 0.0:
        multiplier = multiplier * np.exp(global_band_alpha * (band_signal - 0.5))
    if external_signal is not None and global_external_alpha != 0.0:
        multiplier = multiplier * np.exp(global_external_alpha * (external_signal - 0.5))
    if low_signal is not None and global_low_alpha != 0.0:
        multiplier = multiplier * np.exp(-global_low_alpha * (low_signal - 0.5))
    return np.clip(base_prediction * multiplier, 0.0, None), alpha, scale


def decile_comparison_frame(frame: pd.DataFrame, base_col: str, adjusted_col: str) -> pd.DataFrame:
    base = actual_decile_metrics(frame, base_col).sort_values("actual_decile")
    adjusted = actual_decile_metrics(frame, adjusted_col).sort_values("actual_decile")
    rows = []
    for base_row, adjusted_row in zip(base.to_dict(orient="records"), adjusted.to_dict(orient="records")):
        decile = int(base_row["actual_decile"]) + 1
        rows.append(
            {
                "decile": f"D{decile}",
                "actual_sum": float(base_row["actual_sum"]),
                "base_predicted_sum": float(base_row["predicted_sum"]),
                "adjusted_predicted_sum": float(adjusted_row["predicted_sum"]),
                "base_ratio": float(base_row["unit_ratio_pred_over_actual"]),
                "adjusted_ratio": float(adjusted_row["unit_ratio_pred_over_actual"]),
                "ratio_delta_adjusted_minus_base": float(adjusted_row["unit_ratio_pred_over_actual"] - base_row["unit_ratio_pred_over_actual"]),
                "base_wape": float(base_row["wape"]),
                "adjusted_wape": float(adjusted_row["wape"]),
            }
        )
    total_actual = float(frame["sales_target"].sum())
    rows.append(
        {
            "decile": "Total",
            "actual_sum": total_actual,
            "base_predicted_sum": float(frame[base_col].sum()),
            "adjusted_predicted_sum": float(frame[adjusted_col].sum()),
            "base_ratio": float(frame[base_col].sum() / total_actual) if total_actual > 0 else math.nan,
            "adjusted_ratio": float(frame[adjusted_col].sum() / total_actual) if total_actual > 0 else math.nan,
            "ratio_delta_adjusted_minus_base": float((frame[adjusted_col].sum() - frame[base_col].sum()) / total_actual) if total_actual > 0 else math.nan,
            "base_wape": metric_row(
                frame["sales_target"].to_numpy(dtype=np.float32),
                frame[base_col].to_numpy(dtype=np.float32),
                "base",
            )["wape"],
            "adjusted_wape": metric_row(
                frame["sales_target"].to_numpy(dtype=np.float32),
                frame[adjusted_col].to_numpy(dtype=np.float32),
                "adjusted",
            )["wape"],
        }
    )
    return pd.DataFrame.from_records(rows)


def sip_band_comparison_frame(frame: pd.DataFrame, base_col: str, adjusted_col: str) -> pd.DataFrame:
    base = sip_band_metrics(frame.rename(columns={base_col: "__prediction"}), "__prediction")
    adjusted = sip_band_metrics(frame.rename(columns={adjusted_col: "__prediction"}), "__prediction")
    rows = []
    for band in ("bottom30", "top30", "top10", "target_1", "target_2", "target_3", "target_4", "target_5_plus", "target_3_plus"):
        rows.append(
            {
                "band": band,
                "rows": adjusted[f"{band}_rows"],
                "actual_sum": adjusted[f"{band}_actual_sum"],
                "base_predicted_sum": base[f"{band}_predicted_sum"],
                "adjusted_predicted_sum": adjusted[f"{band}_predicted_sum"],
                "base_ratio": base[f"{band}_ratio"],
                "adjusted_ratio": adjusted[f"{band}_ratio"],
                "ratio_delta_adjusted_minus_base": adjusted[f"{band}_ratio"] - base[f"{band}_ratio"],
            }
        )
    return pd.DataFrame.from_records(rows)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_group_cols(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tail classifier and calibrate a base positive-sales regressor.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["regressor_positive"], default="regressor_positive")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--tail-quantile", type=float, default=0.80)
    parser.add_argument("--tail-threshold", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=350)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-max-depth", type=int, default=5)
    parser.add_argument("--classifier-min-child-weight", type=float, default=50.0)
    parser.add_argument("--classifier-eta", type=float, default=0.05)
    parser.add_argument("--classifier-subsample", type=float, default=0.85)
    parser.add_argument("--classifier-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--classifier-scale-pos-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--multiplier-mode", choices=["rank", "probability"], default="rank")
    parser.add_argument(
        "--selection-metric",
        choices=["composite", "strict", "sip_band", "sip_top_consensus", "sip_top_push", "sip_tail_gain"],
        default="composite",
    )
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--band-alpha-grid", default="0")
    parser.add_argument("--external-alpha-grid", default="0")
    parser.add_argument("--low-alpha-grid", default="0")
    parser.add_argument("--scale-grid", default="0.90,0.95,1.0,1.05,1.10,1.15")
    parser.add_argument("--external-signal-run-dir", type=Path, default=None)
    parser.add_argument("--enable-band-layer", action="store_true")
    parser.add_argument("--enable-low-layer", action="store_true")
    parser.add_argument("--low-target-max", type=float, default=1.0)
    parser.add_argument("--band-lower-quantile", type=float, default=0.60)
    parser.add_argument("--band-upper-quantile", type=float, default=0.80)
    parser.add_argument("--band-target-min", type=float, default=None)
    parser.add_argument("--band-target-max", type=float, default=None)
    parser.add_argument("--calibration-group-cols", default="")
    parser.add_argument("--group-min-rows", type=int, default=2500)
    parser.add_argument("--group-shrinkage", type=float, default=10000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, manifest, contract = load_dataset(args.dataset_dir, args.family, args.stage)
    contract, drop_report = apply_feature_drops(contract, args.family)
    train_df = frame.loc[frame["split"] == "train"].copy()
    valid_df = frame.loc[frame["split"] == "valid"].copy()
    test_df = frame.loc[frame["split"] == "test"].copy()
    artifact = fit_encoder(train_df, contract["numeric_features"], contract["categorical_features"])
    run_dir = (
        args.output_dir
        / args.family.lower()
        / args.stage
        / tail_target_name(args.tail_quantile, args.tail_threshold)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_drop_report.json").write_text(json.dumps(drop_report, indent=2), encoding="utf-8")
    (run_dir / "feature_contract_effective.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    tail_target = tail_target_name(args.tail_quantile, args.tail_threshold)
    train_tail_labels = target_tail_labels(train_df, args.tail_quantile, args.tail_threshold)
    valid_tail_labels = target_tail_labels(valid_df, args.tail_quantile, args.tail_threshold)
    booster, train_prob, valid_prob = train_classifier(
        train_df,
        valid_df,
        artifact,
        y_train=train_tail_labels,
        y_valid=valid_tail_labels,
        label=tail_target,
        rounds=args.rounds,
        early_stopping=args.early_stopping,
        seed=args.seed,
        max_depth=args.classifier_max_depth,
        min_child_weight=args.classifier_min_child_weight,
        eta=args.classifier_eta,
        subsample=args.classifier_subsample,
        colsample_bytree=args.classifier_colsample_bytree,
        scale_pos_weight_multiplier=args.classifier_scale_pos_weight_multiplier,
    )
    test_prob = predict_classifier(booster, test_df, artifact)
    booster.save_model(run_dir / "tail_classifier.json")
    (run_dir / "tail_encoding_artifact.json").write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")

    train_band_prob = None
    valid_band_prob = None
    test_band_prob = None
    band_booster = None
    band_target = band_target_name(
        args.band_lower_quantile,
        args.band_upper_quantile,
        args.band_target_min,
        args.band_target_max,
    )
    if args.enable_band_layer:
        if args.band_target_min is None and args.band_lower_quantile >= args.band_upper_quantile:
            raise ValueError("--band-lower-quantile must be below --band-upper-quantile")
        if args.band_target_max is not None and args.band_target_min is None:
            raise ValueError("--band-target-max requires --band-target-min")
        if args.band_target_max is not None and args.band_target_min > args.band_target_max:
            raise ValueError("--band-target-min must be <= --band-target-max")
        train_band_labels = target_band_labels(
            train_df,
            args.band_lower_quantile,
            args.band_upper_quantile,
            args.band_target_min,
            args.band_target_max,
        )
        valid_band_labels = target_band_labels(
            valid_df,
            args.band_lower_quantile,
            args.band_upper_quantile,
            args.band_target_min,
            args.band_target_max,
        )
        band_booster, train_band_prob, valid_band_prob = train_classifier(
            train_df,
            valid_df,
            artifact,
            y_train=train_band_labels,
            y_valid=valid_band_labels,
            label=band_target,
            rounds=args.rounds,
            early_stopping=args.early_stopping,
            seed=args.seed + 17,
            max_depth=args.classifier_max_depth,
            min_child_weight=args.classifier_min_child_weight,
            eta=args.classifier_eta,
            subsample=args.classifier_subsample,
            colsample_bytree=args.classifier_colsample_bytree,
            scale_pos_weight_multiplier=args.classifier_scale_pos_weight_multiplier,
        )
        test_band_prob = predict_classifier(band_booster, test_df, artifact)
        band_booster.save_model(run_dir / "band_classifier.json")

    train_low_prob = None
    valid_low_prob = None
    test_low_prob = None
    low_booster = None
    low_target = low_target_name(args.low_target_max if args.enable_low_layer else None)
    if args.enable_low_layer:
        train_low_labels = low_target_labels(train_df, args.low_target_max)
        valid_low_labels = low_target_labels(valid_df, args.low_target_max)
        low_booster, train_low_prob, valid_low_prob = train_classifier(
            train_df,
            valid_df,
            artifact,
            y_train=train_low_labels,
            y_valid=valid_low_labels,
            label=low_target,
            rounds=args.rounds,
            early_stopping=args.early_stopping,
            seed=args.seed + 31,
            max_depth=args.classifier_max_depth,
            min_child_weight=args.classifier_min_child_weight,
            eta=args.classifier_eta,
            subsample=args.classifier_subsample,
            colsample_bytree=args.classifier_colsample_bytree,
            scale_pos_weight_multiplier=args.classifier_scale_pos_weight_multiplier,
        )
        test_low_prob = predict_classifier(low_booster, test_df, artifact)
        low_booster.save_model(run_dir / "low_classifier.json")

    base_booster, base_artifact, base_objective = load_base_regressor(args.base_run_dir)
    valid_base_prediction = predict_frame(base_booster, valid_df, base_artifact, base_objective)
    test_base_prediction = predict_frame(base_booster, test_df, base_artifact, base_objective)

    external_objective = None
    valid_external_prediction = None
    test_external_prediction = None
    if args.external_signal_run_dir is not None:
        external_booster, external_artifact, external_objective = load_base_regressor(args.external_signal_run_dir)
        valid_external_prediction = predict_frame(external_booster, valid_df, external_artifact, external_objective)
        test_external_prediction = predict_frame(external_booster, test_df, external_artifact, external_objective)

    grid = calibration_grid(
        valid_df,
        valid_base_prediction,
        valid_prob,
        valid_band_prob,
        valid_external_prediction,
        valid_low_prob,
        mode=args.multiplier_mode,
        alphas=parse_float_list(args.alpha_grid),
        band_alphas=parse_float_list(args.band_alpha_grid) if args.enable_band_layer else [0.0],
        external_alphas=parse_float_list(args.external_alpha_grid) if args.external_signal_run_dir is not None else [0.0],
        low_alphas=parse_float_list(args.low_alpha_grid) if args.enable_low_layer else [0.0],
        scales=parse_float_list(args.scale_grid),
        selection_metric=args.selection_metric,
    )
    grid.to_csv(run_dir / "validation_multiplier_grid.csv", index=False)
    best = grid.iloc[0].to_dict()
    alpha = float(best["alpha"])
    band_alpha = float(best.get("band_alpha", 0.0))
    external_alpha = float(best.get("external_alpha", 0.0))
    low_alpha = float(best.get("low_alpha", 0.0))
    scale = float(best["scale"])
    log(
        "Best validation multiplier: "
        f"alpha={alpha:.3f}, band_alpha={band_alpha:.3f}, external_alpha={external_alpha:.3f}, "
        f"low_alpha={low_alpha:.3f}, scale={scale:.3f}, "
        f"composite_decile_score={best['composite_decile_score']:.4f}, "
        f"strict_composite_decile_score={best['strict_composite_decile_score']:.4f}, "
        f"sip_band_score={best['sip_band_score']:.4f}, "
        f"sip_top_consensus_score={best.get('sip_top_consensus_score', math.nan):.4f}, "
        f"sip_top_push_score={best.get('sip_top_push_score', math.nan):.4f}, "
        f"sip_tail_gain_score={best.get('sip_tail_gain_score', math.nan):.4f}, "
        f"curve_score={best['curve_score']:.4f}"
    )
    group_cols = parse_group_cols(args.calibration_group_cols)
    if args.enable_band_layer and group_cols:
        raise ValueError("Band-layer calibration currently supports global calibration only; use --calibration-group-cols ''")
    group_table = grouped_calibration_table(
        valid_df,
        valid_base_prediction,
        valid_prob,
        mode=args.multiplier_mode,
        group_cols=group_cols,
        alphas=parse_float_list(args.alpha_grid),
        scales=parse_float_list(args.scale_grid),
        global_alpha=alpha,
        global_scale=scale,
        min_rows=args.group_min_rows,
        shrinkage=args.group_shrinkage,
    )
    if not group_table.empty:
        group_table.to_csv(run_dir / "validation_group_multiplier_table.csv", index=False)
        log(f"Using {len(group_table)} group-aware multiplier rows for columns {group_cols}")

    valid_df["base_prediction"] = valid_base_prediction
    valid_df["tail_probability"] = valid_prob
    valid_df["tail_probability_rank"] = probability_rank(valid_prob)
    if valid_band_prob is not None:
        valid_df["band_probability"] = valid_band_prob
        valid_df["band_probability_rank"] = probability_rank(valid_band_prob)
    valid_adjusted, valid_alpha, valid_scale = apply_grouped_adjustment(
        valid_df,
        valid_base_prediction,
        valid_prob,
        valid_band_prob,
        valid_external_prediction,
        valid_low_prob,
        mode=args.multiplier_mode,
        group_cols=group_cols,
        group_table=group_table,
        global_alpha=alpha,
        global_scale=scale,
        global_band_alpha=band_alpha,
        global_external_alpha=external_alpha,
        global_low_alpha=low_alpha,
    )
    valid_df["adjusted_prediction"] = valid_adjusted
    valid_df["multiplier_alpha"] = valid_alpha
    valid_df["multiplier_band_alpha"] = band_alpha
    valid_df["multiplier_external_alpha"] = external_alpha
    valid_df["multiplier_low_alpha"] = low_alpha
    valid_df["multiplier_scale"] = valid_scale
    test_df["base_prediction"] = test_base_prediction
    test_df["tail_probability"] = test_prob
    test_df["tail_probability_rank"] = probability_rank(test_prob)
    if test_band_prob is not None:
        test_df["band_probability"] = test_band_prob
        test_df["band_probability_rank"] = probability_rank(test_band_prob)
    if test_external_prediction is not None:
        test_df["external_signal_prediction"] = test_external_prediction
        test_df["external_signal_rank"] = probability_rank(test_external_prediction)
    if test_low_prob is not None:
        test_df["low_probability"] = test_low_prob
        test_df["low_probability_rank"] = probability_rank(test_low_prob)
    test_adjusted, test_alpha, test_scale = apply_grouped_adjustment(
        test_df,
        test_base_prediction,
        test_prob,
        test_band_prob,
        test_external_prediction,
        test_low_prob,
        mode=args.multiplier_mode,
        group_cols=group_cols,
        group_table=group_table,
        global_alpha=alpha,
        global_scale=scale,
        global_band_alpha=band_alpha,
        global_external_alpha=external_alpha,
        global_low_alpha=low_alpha,
    )
    test_df["adjusted_prediction"] = test_adjusted
    test_df["multiplier_alpha"] = test_alpha
    test_df["multiplier_band_alpha"] = band_alpha
    test_df["multiplier_external_alpha"] = external_alpha
    test_df["multiplier_low_alpha"] = low_alpha
    test_df["multiplier_scale"] = test_scale

    write_diagnostics(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction", run_dir / "final_validation_adjusted", "validation_adjusted")
    write_diagnostics(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction", run_dir / "final_test_adjusted", "test_adjusted")
    validation_comparison = decile_comparison_frame(valid_df, "base_prediction", "adjusted_prediction")
    test_comparison = decile_comparison_frame(test_df, "base_prediction", "adjusted_prediction")
    validation_comparison.to_csv(run_dir / "validation_decile_comparison.csv", index=False)
    test_comparison.to_csv(run_dir / "test_decile_comparison.csv", index=False)
    sip_band_comparison_frame(valid_df, "base_prediction", "adjusted_prediction").to_csv(
        run_dir / "validation_sip_band_comparison.csv",
        index=False,
    )
    sip_band_comparison_frame(test_df, "base_prediction", "adjusted_prediction").to_csv(
        run_dir / "test_sip_band_comparison.csv",
        index=False,
    )
    log_decile_unit_table(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction", "Adjusted final test").to_csv(
        run_dir / "final_test_adjusted" / "test_adjusted_decile_unit_table.csv"
    )
    prediction_columns = [
            "store_id",
            "product_id",
            "onsaledate",
            "sales_target",
            "base_prediction",
            "tail_probability",
            "tail_probability_rank",
    ]
    if test_band_prob is not None:
        prediction_columns.extend(["band_probability", "band_probability_rank"])
    if test_external_prediction is not None:
        prediction_columns.extend(["external_signal_prediction", "external_signal_rank"])
    if test_low_prob is not None:
        prediction_columns.extend(["low_probability", "low_probability_rank"])
    prediction_columns.extend(
        [
            "adjusted_prediction",
            "multiplier_alpha",
            "multiplier_band_alpha",
            "multiplier_external_alpha",
            "multiplier_low_alpha",
            "multiplier_scale",
            "store_chain",
            "classoftrade",
            "segment",
            "subsegment",
            "title",
        ]
    )
    test_df[prediction_columns].to_csv(run_dir / "test_tail_adjusted_predictions.csv", index=False)

    classifier_rows = [
        classifier_metrics(train_df, train_prob, train_tail_labels, tail_target, "train"),
        classifier_metrics(valid_df, valid_prob, valid_tail_labels, tail_target, "validation"),
        classifier_metrics(
            test_df,
            test_prob,
            target_tail_labels(test_df, args.tail_quantile, args.tail_threshold),
            tail_target,
            "test",
        ),
    ]
    if args.enable_band_layer:
        classifier_rows.extend(
            [
                classifier_metrics(
                    train_df,
                    train_band_prob,
                    target_band_labels(
                        train_df,
                        args.band_lower_quantile,
                        args.band_upper_quantile,
                        args.band_target_min,
                        args.band_target_max,
                    ),
                    band_target,
                    "train",
                ),
                classifier_metrics(
                    valid_df,
                    valid_band_prob,
                    target_band_labels(
                        valid_df,
                        args.band_lower_quantile,
                        args.band_upper_quantile,
                        args.band_target_min,
                        args.band_target_max,
                    ),
                    band_target,
                    "validation",
                ),
                classifier_metrics(
                    test_df,
                    test_band_prob,
                    target_band_labels(
                        test_df,
                        args.band_lower_quantile,
                        args.band_upper_quantile,
                        args.band_target_min,
                        args.band_target_max,
                    ),
                    band_target,
                    "test",
                ),
            ]
        )
    if args.enable_low_layer:
        classifier_rows.extend(
            [
                classifier_metrics(train_df, train_low_prob, low_target_labels(train_df, args.low_target_max), low_target, "train"),
                classifier_metrics(valid_df, valid_low_prob, low_target_labels(valid_df, args.low_target_max), low_target, "validation"),
                classifier_metrics(test_df, test_low_prob, low_target_labels(test_df, args.low_target_max), low_target, "test"),
            ]
        )
    classifier_summary = pd.DataFrame.from_records(classifier_rows)
    classifier_summary.to_csv(run_dir / "tail_classifier_metrics.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "tail_quantile": args.tail_quantile,
        "tail_threshold": args.tail_threshold,
        "tail_target": tail_target,
        "base_run_dir": str(args.base_run_dir),
        "base_objective": base_objective,
        "external_signal_run_dir": str(args.external_signal_run_dir) if args.external_signal_run_dir else None,
        "external_signal_objective": external_objective,
        "classifier_params": {
            "max_depth": args.classifier_max_depth,
            "min_child_weight": args.classifier_min_child_weight,
            "eta": args.classifier_eta,
            "subsample": args.classifier_subsample,
            "colsample_bytree": args.classifier_colsample_bytree,
            "scale_pos_weight_multiplier": args.classifier_scale_pos_weight_multiplier,
        },
        "multiplier_mode": args.multiplier_mode,
        "selection_metric": args.selection_metric,
        "band_layer": {
            "enabled": bool(args.enable_band_layer),
            "lower_quantile": float(args.band_lower_quantile),
            "upper_quantile": float(args.band_upper_quantile),
            "target_min": args.band_target_min,
            "target_max": args.band_target_max,
        },
        "low_layer": {
            "enabled": bool(args.enable_low_layer),
            "target_max": args.low_target_max if args.enable_low_layer else None,
            "target": low_target,
        },
        "calibration_group_cols": group_cols,
        "group_min_rows": args.group_min_rows,
        "group_shrinkage": args.group_shrinkage,
        "group_multiplier_rows": int(len(group_table)),
        "best_multiplier": {
            "alpha": alpha,
            "band_alpha": band_alpha,
            "external_alpha": external_alpha,
            "low_alpha": low_alpha,
            "scale": scale,
            "validation_curve_score": float(best["curve_score"]),
            "validation_composite_decile_score": float(best["composite_decile_score"]),
            "validation_strict_composite_decile_score": float(best["strict_composite_decile_score"]),
            "validation_sip_band_score": float(best["sip_band_score"]),
            "validation_sip_top_consensus_score": float(best.get("sip_top_consensus_score", math.nan)),
            "validation_sip_top_push_score": float(best.get("sip_top_push_score", math.nan)),
            "validation_sip_tail_gain_score": float(best.get("sip_tail_gain_score", math.nan)),
        },
        "classifier_metrics": classifier_summary.to_dict(orient="records"),
        "validation_base": metric_row(valid_df["sales_target"].to_numpy(dtype=np.float32), valid_df["base_prediction"].to_numpy(dtype=np.float32), "validation_base"),
        "validation_adjusted": metric_row(valid_df["sales_target"].to_numpy(dtype=np.float32), valid_df["adjusted_prediction"].to_numpy(dtype=np.float32), "validation_adjusted"),
        "test_base": metric_row(test_df["sales_target"].to_numpy(dtype=np.float32), test_df["base_prediction"].to_numpy(dtype=np.float32), "test_base"),
        "test_adjusted": metric_row(test_df["sales_target"].to_numpy(dtype=np.float32), test_df["adjusted_prediction"].to_numpy(dtype=np.float32), "test_adjusted"),
        "validation_adjusted_composite_decile_score": composite_decile_score(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "test_adjusted_composite_decile_score": composite_decile_score(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "validation_adjusted_strict_composite_decile_score": strict_composite_decile_score(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "test_adjusted_strict_composite_decile_score": strict_composite_decile_score(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "validation_adjusted_sip_band_score": sip_band_score(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "test_adjusted_sip_band_score": sip_band_score(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "validation_adjusted_sip_top_consensus_score": sip_top_consensus_score(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "test_adjusted_sip_top_consensus_score": sip_top_consensus_score(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "validation_adjusted_sip_top_push_score": sip_top_push_score(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "test_adjusted_sip_top_push_score": sip_top_push_score(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "validation_adjusted_sip_tail_gain_score": sip_tail_gain_score(valid_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "test_adjusted_sip_tail_gain_score": sip_tail_gain_score(test_df.rename(columns={"adjusted_prediction": "prediction"}), "prediction"),
        "manifest": manifest,
        "feature_drop_report": drop_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote tail-layer outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
