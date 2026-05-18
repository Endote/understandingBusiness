#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from train_incidence_classifier import calibration_table, drop_matching_features, parse_string_list
from train_stage_model import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    apply_feature_drops,
    artifact_payload,
    best_iteration_end,
    best_iteration_value,
    best_score_value,
    fit_encoder,
    load_dataset,
    transform_frame,
)


OBJECTIVE = "amount_band_multiclass"
BAND_COLUMNS = ["band_p0_zero", "band_p1_one", "band_p2_two", "band_p3_three_four", "band_p4_five_plus"]
BAND_VALUES = np.array([0.0, 1.0, 2.0, 3.5, 5.5], dtype=np.float32)
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def band_labels(frame: pd.DataFrame) -> np.ndarray:
    sales = frame["sales_target"].to_numpy(dtype=np.float32)
    labels = np.zeros(len(frame), dtype=np.int32)
    labels[sales == 1.0] = 1
    labels[sales == 2.0] = 2
    labels[(sales >= 3.0) & (sales <= 4.0)] = 3
    labels[sales >= 5.0] = 4
    return labels


def normalize_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim == 1:
        probability = probability.reshape((-1, len(BAND_COLUMNS)))
    if probability.shape[1] != len(BAND_COLUMNS):
        raise ValueError(f"Expected {len(BAND_COLUMNS)} class probabilities, got shape {probability.shape}")
    return probability


def class_weights(labels: np.ndarray, power: float, max_weight: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(BAND_COLUMNS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw = np.power(counts.sum() / (len(counts) * counts), power)
    raw = np.clip(raw, 1.0 / max_weight, max_weight)
    weights = raw[labels]
    return (weights / weights.mean()).astype(np.float32)


def band_metrics(frame: pd.DataFrame, probability: np.ndarray, split: str) -> dict[str, object]:
    probability = normalize_probability(probability)
    labels = band_labels(frame)
    expected_units = probability @ BAND_VALUES
    tail_probability = probability[:, 3] + probability[:, 4]
    positive_probability = 1.0 - probability[:, 0]
    result: dict[str, object] = {
        "split": split,
        "rows": int(len(frame)),
        "log_loss": float(log_loss(labels, np.clip(probability, 1e-7, 1.0), labels=list(range(len(BAND_COLUMNS))))),
        "actual_mean_units": float(frame["sales_target"].mean()),
        "expected_units_mean": float(expected_units.mean()),
        "positive_probability_mean": float(positive_probability.mean()),
        "tail_probability_mean": float(tail_probability.mean()),
    }
    for band_id, col in enumerate(BAND_COLUMNS):
        mask = labels == band_id
        result[f"{col}_rate"] = float(mask.mean())
        result[f"{col}_probability_mean"] = float(probability[:, band_id].mean())
    for top_frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        n = max(1, math.ceil(len(frame) * top_frac))
        expected_idx = np.argsort(expected_units)[-n:]
        tail_idx = np.argsort(tail_probability)[-n:]
        result[f"sales_units_at_expected_top_{int(top_frac * 100)}"] = float(frame.iloc[expected_idx]["sales_target"].sum())
        result[f"tail_ge3_recall_at_expected_top_{int(top_frac * 100)}"] = float(
            (frame.iloc[expected_idx]["sales_target"].to_numpy(dtype=np.float32) >= 3.0).sum()
            / max((frame["sales_target"].to_numpy(dtype=np.float32) >= 3.0).sum(), 1)
        )
        result[f"sales_units_at_tailprob_top_{int(top_frac * 100)}"] = float(frame.iloc[tail_idx]["sales_target"].sum())
        result[f"tail_ge3_recall_at_tailprob_top_{int(top_frac * 100)}"] = float(
            (frame.iloc[tail_idx]["sales_target"].to_numpy(dtype=np.float32) >= 3.0).sum()
            / max((frame["sales_target"].to_numpy(dtype=np.float32) >= 3.0).sum(), 1)
        )
    return result


def write_predictions(path: Path, frame: pd.DataFrame, probability: np.ndarray) -> None:
    probability = normalize_probability(probability)
    out = frame[KEY_COLUMNS + ["split", "sales_target", "positive_sale_flag"]].copy()
    out["onsaledate"] = pd.to_datetime(out["onsaledate"])
    for idx, col in enumerate(BAND_COLUMNS):
        out[col] = probability[:, idx].astype(np.float32)
    out["band_positive_probability"] = (1.0 - probability[:, 0]).astype(np.float32)
    out["band_tail_probability"] = (probability[:, 3] + probability[:, 4]).astype(np.float32)
    out["band_expected_units"] = (probability @ BAND_VALUES).astype(np.float32)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all-row SIP/Weeklies amount-band incidence classifier.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["classifier_all"], default="classifier_all")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=50.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
    parser.add_argument(
        "--drop-feature-substrings",
        default="affinity,embedding_pca_,embedding_analog,_obs",
        help="Comma-separated feature-name substrings to drop after normal feature contract drops.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log(f"Loading {args.family} {args.stage} dataset")
    frame, manifest, contract = load_dataset(args.dataset_dir, args.family, args.stage)
    dataset_rows = int(len(frame))
    contract, drop_report = apply_feature_drops(contract, args.family)
    contract, substring_drop_report = drop_matching_features(contract, parse_string_list(args.drop_feature_substrings))
    log(f"Using {len(contract['numeric_features'])} numeric and {len(contract['categorical_features'])} categorical features")

    train_df = frame.loc[frame["split"] == "train"].copy()
    valid_df = frame.loc[frame["split"] == "valid"].copy()
    test_df = frame.loc[frame["split"] == "test"].copy()
    del frame
    gc.collect()

    artifact = fit_encoder(train_df, contract["numeric_features"], contract["categorical_features"])
    y_train = band_labels(train_df)
    y_valid = band_labels(valid_df)
    weights = class_weights(y_train, power=args.class_weight_power, max_weight=args.max_class_weight)

    log("Encoding train and validation matrices")
    train_encoded = transform_frame(train_df, artifact, "tweedie", "uniform")
    valid_encoded = transform_frame(valid_df, artifact, "tweedie", "uniform")
    dtrain = xgb.DMatrix(train_encoded.matrix, label=y_train, weight=weights, feature_names=artifact.feature_names)
    dvalid = xgb.DMatrix(valid_encoded.matrix, label=y_valid, feature_names=artifact.feature_names)

    run_dir = (
        args.output_dir
        / args.family.lower()
        / args.stage
        / OBJECTIVE
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_drop_report.json").write_text(json.dumps(drop_report, indent=2), encoding="utf-8")
    (run_dir / "feature_substring_drop_report.json").write_text(json.dumps(substring_drop_report, indent=2), encoding="utf-8")
    (run_dir / "feature_contract_effective.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")

    params = {
        "objective": "multi:softprob",
        "num_class": len(BAND_COLUMNS),
        "eval_metric": ["mlogloss"],
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "seed": args.seed,
    }
    log(
        "Training amount-band classifier: "
        f"train={len(train_df)}, valid={len(valid_df)}, class_counts={np.bincount(y_train, minlength=len(BAND_COLUMNS)).tolist()}"
    )
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping if args.early_stopping > 0 else None,
        verbose_eval=25,
    )
    booster.save_model(run_dir / "model.json")
    (run_dir / "feature_importance_gain.json").write_text(json.dumps(booster.get_score(importance_type="gain"), indent=2), encoding="utf-8")

    valid_probability = normalize_probability(booster.predict(dvalid, iteration_range=(0, best_iteration_end(booster))))
    del dtrain, dvalid, train_encoded, valid_encoded
    gc.collect()
    log("Predicting test probabilities")
    test_encoded = transform_frame(test_df, artifact, "tweedie", "uniform")
    dtest = xgb.DMatrix(test_encoded.matrix, feature_names=artifact.feature_names)
    test_probability = normalize_probability(booster.predict(dtest, iteration_range=(0, best_iteration_end(booster))))

    metrics = pd.DataFrame.from_records(
        [
            band_metrics(valid_df, valid_probability, "validation"),
            band_metrics(test_df, test_probability, "test"),
        ]
    )
    metrics.to_csv(run_dir / "amount_band_metrics.csv", index=False)
    calibration_table(valid_df, 1.0 - valid_probability[:, 0], bins=10).to_csv(
        run_dir / "validation_positive_probability_calibration.csv", index=False
    )
    calibration_table(test_df, 1.0 - test_probability[:, 0], bins=10).to_csv(
        run_dir / "test_positive_probability_calibration.csv", index=False
    )
    write_predictions(run_dir / "validation_amount_band_predictions.csv", valid_df, valid_probability)
    write_predictions(run_dir / "test_amount_band_predictions.csv", test_df, test_probability)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "objective": OBJECTIVE,
        "band_columns": BAND_COLUMNS,
        "band_values": BAND_VALUES.tolist(),
        "xgboost_params": params,
        "class_weight_power": args.class_weight_power,
        "max_class_weight": args.max_class_weight,
        "best_iteration": best_iteration_value(booster),
        "best_score": best_score_value(booster),
        "dataset_rows": dataset_rows,
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_drop_report,
        "metrics": metrics.to_dict(orient="records"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote amount-band classifier outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
