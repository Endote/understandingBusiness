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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from train_incidence_classifier import calibration_table, labels, threshold_table
from train_stage_model import DEFAULT_DATASET_DIR, DEFAULT_OUTPUT_DIR, EncodingArtifact, load_dataset, transform_frame


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


def predict_probability(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    end = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1)) + 1
    return booster.predict(dmatrix, iteration_range=(0, end))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def inv_logit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def intercept_for_target_mean(probability: np.ndarray, target_mean: float) -> float:
    target_mean = float(np.clip(target_mean, 1e-6, 1 - 1e-6))
    raw_logit = logit(probability)
    low = -10.0
    high = 10.0
    for _ in range(80):
        mid = (low + high) / 2.0
        mean = float(inv_logit(raw_logit + mid).mean())
        if mean < target_mean:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def probability_bins_from_validation(valid_probability: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(valid_probability, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.maximum.accumulate(edges)


def assign_probability_bin(probability: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], probability, side="right")


def shrunk_multiplier_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
    group_cols: list[str],
    target_overprediction: float,
    shrinkage: float,
    min_rows: int,
    global_multiplier: float,
) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame()
    work = frame[group_cols].astype("string").fillna("UNKNOWN").copy()
    work["positive_sale_flag"] = labels(frame)
    work["probability"] = probability
    rows = []
    for key, group in work.groupby(group_cols, dropna=False, observed=True):
        if len(group) < min_rows:
            continue
        actual_rate = float(group["positive_sale_flag"].mean())
        predicted_mean = float(group["probability"].mean())
        if predicted_mean <= 0:
            continue
        raw_multiplier = target_overprediction * actual_rate / predicted_mean
        shrink = len(group) / (len(group) + shrinkage)
        multiplier = math.exp(math.log(global_multiplier) + shrink * (math.log(max(raw_multiplier, 1e-6)) - math.log(global_multiplier)))
        key_values = key if isinstance(key, tuple) else (key,)
        row = {
            "rows": int(len(group)),
            "actual_positive_rate": actual_rate,
            "predicted_probability_mean": predicted_mean,
            "raw_multiplier": float(raw_multiplier),
            "shrinkage_weight": float(shrink),
            "multiplier": float(multiplier),
        }
        row.update({col: str(value) for col, value in zip(group_cols, key_values)})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def apply_group_multiplier(
    frame: pd.DataFrame,
    probability: np.ndarray,
    group_cols: list[str],
    table: pd.DataFrame,
    global_multiplier: float,
) -> np.ndarray:
    if not group_cols or table.empty:
        return np.clip(probability * global_multiplier, 0.0, 1.0)
    key_frame = frame[group_cols].astype("string").fillna("UNKNOWN").copy()
    key_frame["__row_order"] = np.arange(len(key_frame), dtype=np.int64)
    multiplier_table = table[[*group_cols, "multiplier"]].copy()
    for col in group_cols:
        multiplier_table[col] = multiplier_table[col].astype("string").fillna("UNKNOWN")
    joined = key_frame.merge(multiplier_table, how="left", on=group_cols, sort=False)
    joined = joined.sort_values("__row_order", kind="stable")
    multipliers = joined["multiplier"].fillna(global_multiplier).to_numpy(dtype=np.float64)
    return np.clip(probability * multipliers, 0.0, 1.0)


def variant_metrics(frame: pd.DataFrame, probability: np.ndarray, variant: str, target_overprediction: float) -> dict[str, object]:
    y = labels(frame)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    table = calibration_table(frame, clipped, bins=10)
    ratios = (
        table["predicted_positive_probability_mean"].to_numpy(dtype=np.float64)
        / np.maximum(table["actual_positive_rate"].to_numpy(dtype=np.float64), 1e-6)
    )
    over_target = ratios - target_overprediction
    return {
        "variant": variant,
        "rows": int(len(frame)),
        "actual_positive_rate": float(y.mean()),
        "probability_mean": float(clipped.mean()),
        "mean_pred_to_actual_decile_ratio": float(ratios.mean()),
        "max_pred_to_actual_decile_ratio": float(ratios.max()),
        "min_pred_to_actual_decile_ratio": float(ratios.min()),
        "mean_abs_decile_ratio_gap_to_target": float(np.abs(over_target).mean()),
        "max_abs_decile_ratio_gap_to_target": float(np.abs(over_target).max()),
        "below_target_deciles": int((ratios < target_overprediction).sum()),
        "brier": float(brier_score_loss(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "log_loss": float(log_loss(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "roc_auc": float(roc_auc_score(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "average_precision": float(average_precision_score(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
    }


def add_prediction_bins(frame: pd.DataFrame, probability: np.ndarray, valid_edges: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["probability_bin"] = assign_probability_bin(probability, valid_edges).astype(str)
    return result


@dataclass
class Variant:
    name: str
    valid_probability: np.ndarray
    test_probability: np.ndarray
    payload: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate an all-row incidence classifier and compare variants.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["classifier_all"], default="classifier_all")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--target-overprediction", type=float, default=1.20)
    parser.add_argument("--group-shrinkage", type=float, default=25000.0)
    parser.add_argument("--group-min-rows", type=int, default=5000)
    parser.add_argument("--probability-bins", type=int, default=10)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    log(f"Loading {args.family} {args.stage} validation/test rows")
    frame, manifest, _ = load_dataset(args.dataset_dir, args.family, args.stage)
    valid_df = frame.loc[frame["split"] == "valid"].copy()
    test_df = frame.loc[frame["split"] == "test"].copy()
    del frame

    booster = xgb.Booster()
    booster.load_model(args.classifier_run_dir / "model.json")
    artifact = artifact_from_payload(json.loads((args.classifier_run_dir / "encoding_artifact.json").read_text(encoding="utf-8")))

    log("Scoring raw validation probabilities")
    valid_raw = predict_probability(booster, valid_df, artifact)
    log("Scoring raw test probabilities")
    test_raw = predict_probability(booster, test_df, artifact)

    valid_target_mean = min(args.target_overprediction * float(labels(valid_df).mean()), 0.999)
    global_scale = valid_target_mean / max(float(valid_raw.mean()), 1e-6)
    intercept = intercept_for_target_mean(valid_raw, valid_target_mean)
    probability_edges = probability_bins_from_validation(valid_raw, args.probability_bins)
    valid_with_bin = add_prediction_bins(valid_df, valid_raw, probability_edges)
    test_with_bin = add_prediction_bins(test_df, test_raw, probability_edges)

    variants: list[Variant] = [
        Variant("raw", valid_raw, test_raw, {}),
        Variant(
            f"global_scale_target_{args.target_overprediction:.2f}",
            np.clip(valid_raw * global_scale, 0.0, 1.0),
            np.clip(test_raw * global_scale, 0.0, 1.0),
            {"global_scale": global_scale},
        ),
        Variant(
            f"logit_intercept_target_{args.target_overprediction:.2f}",
            inv_logit(logit(valid_raw) + intercept),
            inv_logit(logit(test_raw) + intercept),
            {"logit_intercept": intercept},
        ),
    ]

    for gamma in (0.75, 0.90, 1.10, 1.25, 1.50):
        valid_power = np.power(np.clip(valid_raw, 1e-6, 1.0), gamma)
        test_power = np.power(np.clip(test_raw, 1e-6, 1.0), gamma)
        scale = valid_target_mean / max(float(valid_power.mean()), 1e-6)
        variants.append(
            Variant(
                f"power_gamma_{gamma:.2f}_scaled_target_{args.target_overprediction:.2f}",
                np.clip(valid_power * scale, 0.0, 1.0),
                np.clip(test_power * scale, 0.0, 1.0),
                {"gamma": gamma, "scale": scale},
            )
        )

    group_specs: dict[str, list[str]] = {
        "probability_bin": ["probability_bin"],
        "title": ["title"],
        "classoftrade": ["classoftrade"],
        "store_chain": ["store_chain"],
        "segment": ["segment"],
        "subsegment": ["subsegment"],
        "title_classoftrade": ["title", "classoftrade"],
        "segment_classoftrade": ["segment", "classoftrade"],
        "probability_bin_classoftrade": ["probability_bin", "classoftrade"],
        "probability_bin_segment": ["probability_bin", "segment"],
        "onsale_month": ["onsale_month_cat"],
        "probability_bin_onsale_month": ["probability_bin", "onsale_month_cat"],
    }
    group_tables: dict[str, pd.DataFrame] = {}
    for name, cols in group_specs.items():
        source_valid = valid_with_bin if "probability_bin" in cols else valid_df
        source_test = test_with_bin if "probability_bin" in cols else test_df
        table = shrunk_multiplier_table(
            source_valid,
            valid_raw,
            cols,
            target_overprediction=args.target_overprediction,
            shrinkage=args.group_shrinkage,
            min_rows=args.group_min_rows,
            global_multiplier=global_scale,
        )
        group_tables[name] = table
        valid_adjusted = apply_group_multiplier(source_valid, valid_raw, cols, table, global_scale)
        test_adjusted = apply_group_multiplier(source_test, test_raw, cols, table, global_scale)
        variants.append(
            Variant(
                f"group_{name}_target_{args.target_overprediction:.2f}",
                valid_adjusted,
                test_adjusted,
                {"group_cols": cols, "rows": int(len(table))},
            )
        )

    run_dir = (
        args.output_dir
        / args.family.lower()
        / args.stage
        / "calibration"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = []
    test_rows = []
    thresholds = parse_float_list(args.thresholds)
    best_name = None
    best_score = math.inf
    for variant in variants:
        validation_metric = variant_metrics(valid_df, variant.valid_probability, variant.name, args.target_overprediction)
        test_metric = variant_metrics(test_df, variant.test_probability, variant.name, args.target_overprediction)
        validation_rows.append(validation_metric)
        test_rows.append(test_metric)
        calibration_table(test_df, variant.test_probability, bins=args.probability_bins).assign(
            pred_to_actual_rate_ratio=lambda df: df["predicted_positive_probability_mean"] / df["actual_positive_rate"].clip(lower=1e-6),
            variant=variant.name,
        ).to_csv(run_dir / f"test_probability_calibration_{variant.name}.csv", index=False)
        threshold_table(test_df, variant.test_probability, thresholds).assign(variant=variant.name).to_csv(
            run_dir / f"test_threshold_metrics_{variant.name}.csv",
            index=False,
        )
        score = (
            validation_metric["mean_abs_decile_ratio_gap_to_target"]
            + 0.30 * validation_metric["max_abs_decile_ratio_gap_to_target"]
            + 0.03 * validation_metric["below_target_deciles"]
        )
        if score < best_score:
            best_score = score
            best_name = variant.name

    validation_summary = pd.DataFrame.from_records(validation_rows).sort_values("mean_abs_decile_ratio_gap_to_target")
    test_summary = pd.DataFrame.from_records(test_rows).sort_values("mean_abs_decile_ratio_gap_to_target")
    validation_summary.to_csv(run_dir / "validation_calibration_variant_summary.csv", index=False)
    test_summary.to_csv(run_dir / "test_calibration_variant_summary.csv", index=False)
    for name, table in group_tables.items():
        table.to_csv(run_dir / f"group_multiplier_table_{name}.csv", index=False)

    selected = next(variant for variant in variants if variant.name == best_name)
    valid_df.assign(raw_probability=valid_raw, calibrated_probability=selected.valid_probability).to_csv(
        run_dir / "validation_calibrated_predictions.csv",
        index=False,
    )
    test_df.assign(raw_probability=test_raw, calibrated_probability=selected.test_probability).to_csv(
        run_dir / "test_calibrated_predictions.csv",
        index=False,
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "classifier_run_dir": str(args.classifier_run_dir),
        "target_overprediction": args.target_overprediction,
        "selected_variant": best_name,
        "selected_variant_payload": selected.payload,
        "validation_selection_score": best_score,
        "manifest": manifest,
        "validation_summary": validation_summary.to_dict(orient="records"),
        "test_summary": test_summary.to_dict(orient="records"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote incidence calibration outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
