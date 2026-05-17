#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

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


OBJECTIVE = "binary_logistic"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["positive_sale_flag"].to_numpy(dtype=np.float32)


def classifier_params(
    seed: int,
    max_depth: int,
    min_child_weight: float,
    eta: float,
    subsample: float,
    colsample_bytree: float,
    scale_pos_weight: float,
) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr", "auc"],
        "tree_method": "hist",
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
    }


def predict_classifier(booster: xgb.Booster, frame: pd.DataFrame, artifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    return booster.predict(dmatrix, iteration_range=(0, best_iteration_end(booster)))


def binary_metrics(frame: pd.DataFrame, probability: np.ndarray, split: str) -> dict[str, object]:
    y = labels(frame)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    result = {
        "split": split,
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()) if len(y) else math.nan,
        "probability_mean": float(probability.mean()) if len(probability) else math.nan,
        "probability_p10": float(np.quantile(probability, 0.10)) if len(probability) else math.nan,
        "probability_p50": float(np.quantile(probability, 0.50)) if len(probability) else math.nan,
        "probability_p90": float(np.quantile(probability, 0.90)) if len(probability) else math.nan,
        "brier": float(brier_score_loss(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "log_loss": float(log_loss(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else math.nan,
        "average_precision": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else math.nan,
    }
    for top_frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        n = max(1, math.ceil(len(frame) * top_frac))
        idx = np.argsort(probability)[-n:]
        result[f"positive_recall_at_pred_top_{int(top_frac * 100)}"] = float(y[idx].sum() / max(y.sum(), 1.0))
        result[f"positive_rate_at_pred_top_{int(top_frac * 100)}"] = float(y[idx].mean())
        result[f"sales_units_at_pred_top_{int(top_frac * 100)}"] = float(frame.iloc[idx]["sales_target"].sum())
    return result


def calibration_table(frame: pd.DataFrame, probability: np.ndarray, bins: int) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "positive_sale_flag": labels(frame),
            "probability": probability,
            "sales_target": frame["sales_target"].to_numpy(dtype=np.float32),
        }
    )
    work["probability_bin"] = pd.qcut(
        work["probability"].rank(method="first"),
        q=min(bins, len(work)),
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bin_id, group in work.groupby("probability_bin", observed=True):
        rows.append(
            {
                "probability_bin": int(bin_id),
                "rows": int(len(group)),
                "predicted_positive_probability_mean": float(group["probability"].mean()),
                "actual_positive_rate": float(group["positive_sale_flag"].mean()),
                "positive_count": int(group["positive_sale_flag"].sum()),
                "actual_sales_units": float(group["sales_target"].sum()),
                "actual_sales_mean": float(group["sales_target"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def threshold_table(frame: pd.DataFrame, probability: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    y = labels(frame)
    rows = []
    for threshold in thresholds:
        pred = probability >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            pred.astype(np.float32),
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": threshold,
                "predicted_positive_rate": float(pred.mean()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive_rows": int(pred.sum()),
                "actual_positive_rows_captured": int(y[pred].sum()) if pred.any() else 0,
                "sales_units_captured": float(frame.loc[pred, "sales_target"].sum()) if pred.any() else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_string_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def drop_matching_features(contract: dict[str, object], substrings: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    if not substrings:
        return contract, {
            "drop_feature_substrings": [],
            "dropped_numeric_features": [],
            "dropped_categorical_features": [],
        }
    numeric = list(contract["numeric_features"])
    categorical = list(contract["categorical_features"])
    dropped_numeric = [feature for feature in numeric if any(token in feature for token in substrings)]
    dropped_categorical = [feature for feature in categorical if any(token in feature for token in substrings)]
    updated = dict(contract)
    updated["numeric_features"] = [feature for feature in numeric if feature not in set(dropped_numeric)]
    updated["categorical_features"] = [feature for feature in categorical if feature not in set(dropped_categorical)]
    return updated, {
        "drop_feature_substrings": substrings,
        "dropped_numeric_features": dropped_numeric,
        "dropped_categorical_features": dropped_categorical,
        "numeric_features_after_substring_drop": len(updated["numeric_features"]),
        "categorical_features_after_substring_drop": len(updated["categorical_features"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all-row positive-sale incidence classifier.")
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
    parser.add_argument("--scale-pos-weight", type=float, default=0.0)
    parser.add_argument(
        "--drop-feature-substrings",
        default="affinity,embedding_pca_",
        help="Comma-separated feature-name substrings to drop after the normal feature contract drops.",
    )
    parser.add_argument("--skip-train-metrics", action="store_true")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log(f"Loading {args.family} {args.stage} dataset")
    frame, manifest, contract = load_dataset(args.dataset_dir, args.family, args.stage)
    dataset_rows = int(len(frame))
    contract, drop_report = apply_feature_drops(contract, args.family)
    contract, substring_drop_report = drop_matching_features(contract, parse_string_list(args.drop_feature_substrings))
    log(
        f"Using {len(contract['numeric_features'])} numeric and {len(contract['categorical_features'])} categorical features "
        f"after substring feature drops"
    )
    train_df = frame.loc[frame["split"] == "train"].copy()
    valid_df = frame.loc[frame["split"] == "valid"].copy()
    test_df = frame.loc[frame["split"] == "test"].copy()
    del frame
    gc.collect()
    log(f"Split rows: train={len(train_df)}, validation={len(valid_df)}, test={len(test_df)}")
    artifact = fit_encoder(train_df, contract["numeric_features"], contract["categorical_features"])

    y_train = labels(train_df)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = args.scale_pos_weight if args.scale_pos_weight > 0 else negatives / positives

    log("Encoding train and validation matrices")
    train_encoded = transform_frame(train_df, artifact, "tweedie", "uniform")
    valid_encoded = transform_frame(valid_df, artifact, "tweedie", "uniform")
    dtrain = xgb.DMatrix(train_encoded.matrix, label=y_train, feature_names=artifact.feature_names)
    dvalid = xgb.DMatrix(valid_encoded.matrix, label=labels(valid_df), feature_names=artifact.feature_names)

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

    log(
        "Training incidence classifier: "
        f"{len(train_df)} train rows, {len(valid_df)} validation rows, "
        f"train positive rate {positives / max(len(train_df), 1):.4f}, "
        f"scale_pos_weight {scale_pos_weight:.3f}"
    )
    booster = xgb.train(
        params=classifier_params(
            seed=args.seed,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            eta=args.eta,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
        ),
        dtrain=dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping if args.early_stopping > 0 else None,
        verbose_eval=25,
    )
    booster.save_model(run_dir / "model.json")
    importance = booster.get_score(importance_type="gain")
    (run_dir / "feature_importance_gain.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")

    train_probability = None if args.skip_train_metrics else booster.predict(dtrain, iteration_range=(0, best_iteration_end(booster)))
    valid_probability = booster.predict(dvalid, iteration_range=(0, best_iteration_end(booster)))
    del dtrain, dvalid, train_encoded, valid_encoded
    gc.collect()
    log("Predicting test probabilities")
    test_probability = predict_classifier(booster, test_df, artifact)
    thresholds = parse_float_list(args.thresholds)

    metric_rows = []
    if train_probability is not None:
        metric_rows.append(binary_metrics(train_df, train_probability, "train"))
    metric_rows.extend(
        [
            binary_metrics(valid_df, valid_probability, "validation"),
            binary_metrics(test_df, test_probability, "test"),
        ]
    )
    metrics = pd.DataFrame.from_records(metric_rows)
    metrics.to_csv(run_dir / "classification_metrics.csv", index=False)
    calibration_table(valid_df, valid_probability, bins=10).to_csv(run_dir / "validation_probability_calibration.csv", index=False)
    calibration_table(test_df, test_probability, bins=10).to_csv(run_dir / "test_probability_calibration.csv", index=False)
    threshold_table(valid_df, valid_probability, thresholds).to_csv(run_dir / "validation_threshold_metrics.csv", index=False)
    threshold_table(test_df, test_probability, thresholds).to_csv(run_dir / "test_threshold_metrics.csv", index=False)

    test_df = test_df.copy()
    test_df["positive_probability"] = test_probability
    test_df[
        [
            "store_id",
            "product_id",
            "onsaledate",
            "sales_target",
            "positive_sale_flag",
            "positive_probability",
            "store_chain",
            "classoftrade",
            "segment",
            "subsegment",
            "title",
        ]
    ].to_csv(run_dir / "test_predictions.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "objective": OBJECTIVE,
        "xgboost_params": {
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "eta": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
        },
        "best_iteration": best_iteration_value(booster),
        "best_score": best_score_value(booster),
        "dataset_rows": dataset_rows,
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_drop_report,
        "metrics": metrics.to_dict(orient="records"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote incidence classifier outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
