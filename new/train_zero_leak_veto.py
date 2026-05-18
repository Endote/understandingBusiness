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

from train_incidence_classifier import drop_matching_features, parse_string_list
from train_stage_model import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    EncodingArtifact,
    apply_feature_drops,
    artifact_payload,
    best_iteration_end,
    best_iteration_value,
    best_score_value,
    fit_encoder,
    load_dataset,
    transform_frame,
)
from train_tail_layer import artifact_from_payload


OBJECTIVE = "zero_leak_veto"
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_encoding(path: Path) -> EncodingArtifact:
    return artifact_from_payload(load_json(path))


def predict_probability(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    return booster.predict(dmatrix, iteration_range=(0, best_iteration_end(booster))).astype(np.float32)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return (frame["sales_target"].to_numpy(dtype=np.float32) <= 0.0).astype(np.float32)


def veto_metrics(frame: pd.DataFrame, probability: np.ndarray, split: str, candidate_threshold: float) -> dict[str, object]:
    y = labels(frame)
    candidate = frame["raw_probability"].to_numpy(dtype=np.float32) >= candidate_threshold
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    result = {
        "split": split,
        "rows": int(len(frame)),
        "zero_rate": float(y.mean()),
        "candidate_rows": int(candidate.sum()),
        "candidate_zero_rate": float(y[candidate].mean()) if candidate.any() else math.nan,
        "probability_mean": float(probability.mean()),
        "candidate_probability_mean": float(probability[candidate].mean()) if candidate.any() else math.nan,
        "log_loss": float(log_loss(y, clipped)) if len(np.unique(y)) > 1 else math.nan,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else math.nan,
        "average_precision": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else math.nan,
    }
    for top_frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        n = max(1, math.ceil(len(frame) * top_frac))
        idx = np.argsort(probability)[-n:]
        result[f"zero_recall_at_veto_top_{int(top_frac * 100)}"] = float(y[idx].sum() / max(y.sum(), 1.0))
        result[f"zero_rate_at_veto_top_{int(top_frac * 100)}"] = float(y[idx].mean())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a zero-leak veto model for SIP/Weeklies incidence candidate rows.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["classifier_all"], default="classifier_all")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incidence-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.675)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--early-stopping", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=35.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
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

    incidence_booster = xgb.Booster()
    incidence_booster.load_model(args.incidence_run_dir / "model.json")
    incidence_artifact = load_encoding(args.incidence_run_dir / "encoding_artifact.json")

    raw_parts = []
    split_frames = {}
    for split in ("train", "valid", "test"):
        split_df = frame.loc[frame["split"] == split].copy()
        log(f"Scoring base incidence raw probability for {split}: {len(split_df)} rows")
        split_df["raw_probability"] = predict_probability(incidence_booster, split_df, incidence_artifact)
        split_frames[split] = split_df
        raw_parts.append(split_df[KEY_COLUMNS + ["raw_probability"]])
    del frame
    gc.collect()

    train_df = split_frames["train"]
    valid_df = split_frames["valid"]
    test_df = split_frames["test"]
    candidate_train = train_df.loc[train_df["raw_probability"] >= args.candidate_threshold].copy()
    candidate_valid = valid_df.loc[valid_df["raw_probability"] >= args.candidate_threshold].copy()
    log(
        "Candidate veto rows: "
        f"train={len(candidate_train)}, valid={len(candidate_valid)}, "
        f"train zero rate={labels(candidate_train).mean():.4f}"
    )

    effective_contract = dict(contract)
    numeric_features = list(effective_contract["numeric_features"])
    if "raw_probability" not in numeric_features:
        numeric_features.append("raw_probability")
    effective_contract["numeric_features"] = numeric_features

    artifact = fit_encoder(candidate_train, effective_contract["numeric_features"], effective_contract["categorical_features"])
    y_train = labels(candidate_train)
    y_valid = labels(candidate_valid)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0

    log("Encoding veto train and validation matrices")
    train_encoded = transform_frame(candidate_train, artifact, "tweedie", "uniform")
    valid_encoded = transform_frame(candidate_valid, artifact, "tweedie", "uniform")
    dtrain = xgb.DMatrix(train_encoded.matrix, label=y_train, feature_names=artifact.feature_names)
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
    (run_dir / "feature_contract_effective.json").write_text(json.dumps(effective_contract, indent=2), encoding="utf-8")
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")

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
    log(f"Training zero-leak veto: scale_pos_weight={scale_pos_weight:.3f}")
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

    del dtrain, dvalid, train_encoded, valid_encoded, candidate_train, candidate_valid
    gc.collect()

    metrics = []
    for split, split_df in (("validation", valid_df), ("test", test_df)):
        log(f"Scoring veto probabilities for {split}")
        encoded = transform_frame(split_df, artifact, "tweedie", "uniform")
        dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
        probability = booster.predict(dmatrix, iteration_range=(0, best_iteration_end(booster))).astype(np.float32)
        metrics.append(veto_metrics(split_df, probability, split, args.candidate_threshold))
        out = split_df[KEY_COLUMNS + ["split", "sales_target", "positive_sale_flag", "raw_probability"]].copy()
        out["veto_zero_probability"] = probability
        out.to_csv(run_dir / f"{split}_veto_predictions.csv", index=False)

    metric_frame = pd.DataFrame.from_records(metrics)
    metric_frame.to_csv(run_dir / "veto_metrics.csv", index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "objective": OBJECTIVE,
        "incidence_run_dir": str(args.incidence_run_dir),
        "candidate_threshold": args.candidate_threshold,
        "xgboost_params": params,
        "best_iteration": best_iteration_value(booster),
        "best_score": best_score_value(booster),
        "dataset_rows": dataset_rows,
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_drop_report,
        "metrics": metric_frame.to_dict(orient="records"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote zero-leak veto outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
