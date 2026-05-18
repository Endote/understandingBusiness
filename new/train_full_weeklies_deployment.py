#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from train_stage_model import (
    ASYM_CURVE_LOG1P_OBJECTIVE,
    CurveObjectiveConfig,
    EncodingArtifact,
    apply_feature_drops,
    artifact_payload,
    asym_curve_log1p_metric,
    asym_curve_log1p_objective,
    fit_encoder,
    load_dataset,
    predict_frame,
    sample_weights,
    transform_frame,
    xgb_params,
)
from train_tail_layer import low_target_labels, target_tail_labels


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = NEW_DIR / "output" / "modeling_datasets"
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "JulyScoring" / "weeklies_full_deployment_models"
REFERENCE_AMOUNT_RUN = NEW_DIR / "output" / "model_runs" / "weeklies" / "regressor_positive" / "asym_curve_log1p" / "20260517_205934"
REFERENCE_TAIL_RUN = NEW_DIR / "output" / "tail_layer_runs" / "weeklies" / "regressor_positive" / "tail_q70" / "20260517_210107"
REFERENCE_CLASSIFIER_RUN = NEW_DIR / "output" / "model_runs" / "weeklies" / "classifier_all" / "binary_logistic" / "20260517_211136"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_rounds(model_path: Path) -> int:
    booster = xgb.Booster()
    booster.load_model(model_path)
    best_iteration = getattr(booster, "best_iteration", None)
    return int(best_iteration) + 1 if best_iteration is not None else booster.num_boosted_rounds()


def train_full_amount(
    dataset_dir: Path,
    output_dir: Path,
    rounds: int,
) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "Weeklies", "regressor_positive")
    contract, drop_report = apply_feature_drops(contract, "Weeklies")
    reference_summary = load_json(REFERENCE_AMOUNT_RUN / "run_summary.json")
    curve_config = CurveObjectiveConfig(**{
        key: value
        for key, value in reference_summary["curve_objective_config"].items()
        if key not in {"prediction_log_clip_min", "prediction_log_clip_max"}
    })
    params = reference_summary["xgboost_params"]
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    encoded = transform_frame(frame, artifact, ASYM_CURVE_LOG1P_OBJECTIVE, reference_summary["weighting"])
    dtrain = xgb.DMatrix(
        encoded.matrix,
        label=encoded.labels,
        weight=encoded.weights,
        feature_names=artifact.feature_names,
    )
    stockout_flags = {id(dtrain): frame.get("stockout_proxy_flag", pd.Series(0, index=frame.index)).to_numpy(dtype=bool)}
    base_score = float(np.mean(encoded.labels)) if len(encoded.labels) else 0.0
    log(f"Training full-label amount regressor on {len(frame)} positive rows for {rounds} rounds")
    booster = xgb.train(
        params=xgb_params(
            ASYM_CURVE_LOG1P_OBJECTIVE,
            seed=42,
            base_score=base_score,
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            eta=float(params["eta"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
        ),
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train")],
        obj=asym_curve_log1p_objective(curve_config, stockout_flags),
        custom_metric=asym_curve_log1p_metric(curve_config),
        maximize=False,
        verbose_eval=10,
    )
    run_dir = output_dir / "regressor_positive"
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(run_dir / "model.json")
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "Weeklies",
        "stage": "regressor_positive",
        "training_scope": "all_labeled_positive_rows",
        "rows": int(len(frame)),
        "objective": ASYM_CURVE_LOG1P_OBJECTIVE,
        "weighting": reference_summary["weighting"],
        "rounds": rounds,
        "xgboost_params": params,
        "curve_objective_config": reference_summary["curve_objective_config"],
        "reference_run": str(REFERENCE_AMOUNT_RUN),
        "manifest": manifest,
        "feature_drop_report": drop_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "rounds": rounds}


def train_binary_full(
    frame: pd.DataFrame,
    artifact: EncodingArtifact,
    y: np.ndarray,
    params: dict[str, object],
    rounds: int,
    model_name: str,
    output_path: Path,
    scale_pos_weight_multiplier: float = 1.0,
) -> dict[str, object]:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    scale_pos_weight = (negatives / positives if positives > 0 else 1.0) * scale_pos_weight_multiplier
    dtrain = xgb.DMatrix(encoded.matrix, label=y, feature_names=artifact.feature_names)
    train_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist",
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "eta": float(params["eta"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
    }
    log(f"Training {model_name} on {len(frame)} rows for {rounds} rounds, positive_rate={positives / max(len(y), 1):.4f}")
    booster = xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train")],
        verbose_eval=25,
    )
    booster.save_model(output_path)
    return {
        "rows": int(len(frame)),
        "positive_rate": float(positives / max(len(y), 1)),
        "rounds": int(rounds),
        "xgboost_params": train_params,
    }


def train_full_tail_layer(dataset_dir: Path, amount_run_dir: Path, output_dir: Path) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "Weeklies", "regressor_positive")
    contract, drop_report = apply_feature_drops(contract, "Weeklies")
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    reference_summary = load_json(REFERENCE_TAIL_RUN / "run_summary.json")
    params = reference_summary["classifier_params"]
    run_dir = output_dir / "tail_layer"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tail_encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    tail_rounds = best_rounds(REFERENCE_TAIL_RUN / "tail_classifier.json")
    low_rounds = best_rounds(REFERENCE_TAIL_RUN / "low_classifier.json")
    tail_y = target_tail_labels(frame, float(reference_summary["tail_quantile"]), reference_summary["tail_threshold"])
    low_y = low_target_labels(frame, float(reference_summary["low_layer"]["target_max"]))
    tail_info = train_binary_full(
        frame,
        artifact,
        tail_y,
        params,
        tail_rounds,
        "full-label tail classifier",
        run_dir / "tail_classifier.json",
        scale_pos_weight_multiplier=float(params.get("scale_pos_weight_multiplier", 1.0)),
    )
    low_info = train_binary_full(
        frame,
        artifact,
        low_y,
        params,
        low_rounds,
        "full-label low classifier",
        run_dir / "low_classifier.json",
        scale_pos_weight_multiplier=float(params.get("scale_pos_weight_multiplier", 1.0)),
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "Weeklies",
        "stage": "regressor_positive",
        "training_scope": "all_labeled_positive_rows",
        "base_run_dir": str(amount_run_dir),
        "base_objective": ASYM_CURVE_LOG1P_OBJECTIVE,
        "tail_quantile": reference_summary["tail_quantile"],
        "tail_threshold": reference_summary["tail_threshold"],
        "tail_target": reference_summary["tail_target"],
        "low_layer": reference_summary["low_layer"],
        "band_layer": {"enabled": False},
        "classifier_params": params,
        "multiplier_mode": reference_summary["multiplier_mode"],
        "best_multiplier": {
            key: reference_summary["best_multiplier"][key]
            for key in ["alpha", "band_alpha", "external_alpha", "low_alpha", "scale"]
        },
        "tail_classifier": tail_info,
        "low_classifier": low_info,
        "reference_run": str(REFERENCE_TAIL_RUN),
        "manifest": manifest,
        "feature_drop_report": drop_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "tail_rounds": tail_rounds, "low_rounds": low_rounds, "rows": int(len(frame))}


def train_full_incidence(dataset_dir: Path, output_dir: Path) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "Weeklies", "classifier_all")
    contract, drop_report = apply_feature_drops(contract, "Weeklies")
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    reference_summary = load_json(REFERENCE_CLASSIFIER_RUN / "run_summary.json")
    params = reference_summary["xgboost_params"]
    run_dir = output_dir / "classifier_all"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    y = frame["positive_sale_flag"].to_numpy(dtype=np.float32)
    rounds = best_rounds(REFERENCE_CLASSIFIER_RUN / "model.json")
    info = train_binary_full(
        frame,
        artifact,
        y,
        params,
        rounds,
        "full-label incidence classifier",
        run_dir / "model.json",
        scale_pos_weight_multiplier=1.0,
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "Weeklies",
        "stage": "classifier_all",
        "training_scope": "all_labeled_rows",
        "objective": "binary_logistic",
        "rounds": rounds,
        "reference_run": str(REFERENCE_CLASSIFIER_RUN),
        "xgboost_params": info["xgboost_params"],
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "manifest": manifest,
        "feature_drop_report": drop_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "rounds": rounds, "positive_rate": float(y.mean())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Weeklies deployment artifacts on all currently labeled rows.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--amount-rounds", type=int, default=best_rounds(REFERENCE_AMOUNT_RUN / "model.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    amount = train_full_amount(args.dataset_dir, run_dir / "models", args.amount_rounds)
    tail = train_full_tail_layer(args.dataset_dir, Path(amount["run_dir"]), run_dir / "models")
    incidence = train_full_incidence(args.dataset_dir, run_dir / "models")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "training_scope": "all currently labeled Weeklies rows",
        "amount": amount,
        "tail": tail,
        "incidence": incidence,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote full-label deployment models to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
