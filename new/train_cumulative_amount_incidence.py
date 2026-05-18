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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

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


OBJECTIVE = "cumulative_amount_incidence"
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def label_for(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return (frame["sales_target"].to_numpy(dtype=np.float32) >= threshold).astype(np.float32)


def threshold_name(threshold: float) -> str:
    if float(threshold).is_integer():
        return f"ge{int(threshold)}"
    return f"ge{str(threshold).replace('.', 'p')}"


def binary_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float, split: str) -> dict[str, object]:
    y = label_for(frame, threshold)
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    result: dict[str, object] = {
        "split": split,
        "threshold": threshold,
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "probability_mean": float(probability.mean()),
        "log_loss": float(log_loss(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else math.nan,
        "average_precision": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else math.nan,
    }
    actual_units = frame["sales_target"].to_numpy(dtype=np.float32)
    for top_frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        n = max(1, math.ceil(len(frame) * top_frac))
        idx = np.argsort(probability)[-n:]
        result[f"positive_recall_at_top_{int(top_frac * 100)}"] = float(y[idx].sum() / max(y.sum(), 1.0))
        result[f"positive_rate_at_top_{int(top_frac * 100)}"] = float(y[idx].mean())
        result[f"sales_units_at_top_{int(top_frac * 100)}"] = float(actual_units[idx].sum())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cumulative all-row amount incidence models P(sales >= k).")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["classifier_all"], default="classifier_all")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=50.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--max-scale-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--drop-feature-substrings",
        default="affinity,embedding_pca_,embedding_analog,_obs",
        help="Comma-separated feature-name substrings to drop after normal feature contract drops.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
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
    log("Encoding train and validation matrices")
    train_encoded = transform_frame(train_df, artifact, "tweedie", "uniform")
    valid_encoded = transform_frame(valid_df, artifact, "tweedie", "uniform")

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

    valid_out = valid_df[KEY_COLUMNS + ["split", "sales_target", "positive_sale_flag"]].copy()
    test_out = test_df[KEY_COLUMNS + ["split", "sales_target", "positive_sale_flag"]].copy()
    metrics: list[dict[str, object]] = []
    model_summaries: list[dict[str, object]] = []

    for threshold in thresholds:
        name = threshold_name(threshold)
        y_train = label_for(train_df, threshold)
        y_valid = label_for(valid_df, threshold)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        scale_pos_weight = min(args.max_scale_pos_weight, negatives / positives) if positives > 0 else 1.0
        log(
            f"Training cumulative {name}: train_positive_rate={y_train.mean():.5f}, "
            f"valid_positive_rate={y_valid.mean():.5f}, scale_pos_weight={scale_pos_weight:.3f}"
        )
        dtrain = xgb.DMatrix(train_encoded.matrix, label=y_train, feature_names=artifact.feature_names)
        dvalid = xgb.DMatrix(valid_encoded.matrix, label=y_valid, feature_names=artifact.feature_names)
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr", "auc"],
            "tree_method": "hist",
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "eta": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "seed": args.seed,
        }
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=args.early_stopping if args.early_stopping > 0 else None,
            verbose_eval=25,
        )
        booster.save_model(run_dir / f"model_{name}.json")
        (run_dir / f"feature_importance_gain_{name}.json").write_text(
            json.dumps(booster.get_score(importance_type="gain"), indent=2),
            encoding="utf-8",
        )
        valid_probability = booster.predict(dvalid, iteration_range=(0, best_iteration_end(booster))).astype(np.float32)
        valid_out[f"cum_{name}_probability"] = valid_probability
        metrics.append(binary_metrics(valid_df, valid_probability, threshold, "validation"))
        calibration_table(valid_df, valid_probability, bins=10).to_csv(
            run_dir / f"validation_{name}_calibration.csv",
            index=False,
        )
        model_summaries.append(
            {
                "threshold": threshold,
                "name": name,
                "xgboost_params": params,
                "best_iteration": best_iteration_value(booster),
                "best_score": best_score_value(booster),
            }
        )
        del dtrain, dvalid
        gc.collect()

        log(f"Predicting test probabilities for cumulative {name}")
        test_encoded = transform_frame(test_df, artifact, "tweedie", "uniform")
        dtest = xgb.DMatrix(test_encoded.matrix, feature_names=artifact.feature_names)
        test_probability = booster.predict(dtest, iteration_range=(0, best_iteration_end(booster))).astype(np.float32)
        test_out[f"cum_{name}_probability"] = test_probability
        metrics.append(binary_metrics(test_df, test_probability, threshold, "test"))
        calibration_table(test_df, test_probability, bins=10).to_csv(run_dir / f"test_{name}_calibration.csv", index=False)
        del test_encoded, dtest, booster
        gc.collect()

    valid_out.to_csv(run_dir / "validation_cumulative_predictions.csv", index=False)
    test_out.to_csv(run_dir / "test_cumulative_predictions.csv", index=False)
    metric_frame = pd.DataFrame.from_records(metrics)
    metric_frame.to_csv(run_dir / "cumulative_metrics.csv", index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "objective": OBJECTIVE,
        "thresholds": thresholds,
        "dataset_rows": dataset_rows,
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_drop_report,
        "models": model_summaries,
        "metrics": metric_frame.to_dict(orient="records"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote cumulative amount incidence outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
