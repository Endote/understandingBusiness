#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

from train_stage_model import (
    DEFAULT_DATASET_DIR,
    apply_feature_drops,
    artifact_payload,
    fit_encoder,
    load_dataset,
    transform_frame,
)


NEW_DIR = Path(__file__).resolve().parent
REFERENCE_CLASSIFIER_RUN = NEW_DIR / "output" / "model_runs" / "weeklies" / "classifier_all" / "binary_logistic" / "20260517_211136"
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "JulyScoring" / "weeklies_full_deployment_models" / "incidence_pruned"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_rounds(model_path: Path) -> int:
    booster = xgb.Booster()
    booster.load_model(model_path)
    best_iteration = getattr(booster, "best_iteration", None)
    return int(best_iteration) + 1 if best_iteration is not None else booster.num_boosted_rounds()


def selected_numeric_features(numeric_features: list[str], top_n: int) -> list[str]:
    importance = load_json(REFERENCE_CLASSIFIER_RUN / "feature_importance_gain.json")
    numeric_set = set(numeric_features)
    ranked = [name for name, _ in sorted(importance.items(), key=lambda kv: kv[1], reverse=True) if name in numeric_set]
    selected = ranked[:top_n]
    for fallback in ["price", "issue_length_days", "onsale_month", "onsale_dow", "merchandised", "facings", "pockets"]:
        if fallback in numeric_set and fallback not in selected:
            selected.append(fallback)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full-label Weeklies incidence classifier with pruned high-gain features.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-numeric", type=int, default=160)
    parser.add_argument("--rounds", type=int, default=best_rounds(REFERENCE_CLASSIFIER_RUN / "model.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, manifest, contract = load_dataset(args.dataset_dir, "Weeklies", "classifier_all")
    contract, drop_report = apply_feature_drops(contract, "Weeklies")
    numeric = selected_numeric_features(list(contract["numeric_features"]), args.top_numeric)
    categorical = list(contract["categorical_features"])
    reduced_contract = dict(contract)
    reduced_contract["numeric_features"] = numeric
    reduced_contract["categorical_features"] = categorical
    artifact = fit_encoder(frame, numeric, categorical)
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    y = frame["positive_sale_flag"].to_numpy(dtype=np.float32)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    reference_summary = load_json(REFERENCE_CLASSIFIER_RUN / "run_summary.json")
    ref_params = reference_summary["xgboost_params"]
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr", "auc"],
        "tree_method": "hist",
        "max_depth": int(ref_params["max_depth"]),
        "min_child_weight": float(ref_params["min_child_weight"]),
        "eta": float(ref_params["eta"]),
        "subsample": float(ref_params["subsample"]),
        "colsample_bytree": float(ref_params["colsample_bytree"]),
        "scale_pos_weight": negatives / positives if positives > 0 else 1.0,
        "seed": 42,
    }
    dtrain = xgb.DMatrix(encoded.matrix, label=y, feature_names=artifact.feature_names)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"Training pruned full-label incidence classifier on {len(frame)} rows, "
        f"{len(numeric)} numeric + {len(categorical)} categorical features, {args.rounds} rounds"
    )
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train")],
        verbose_eval=25,
    )
    booster.save_model(run_dir / "model.json")
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "Weeklies",
        "stage": "classifier_all",
        "training_scope": "all_labeled_rows",
        "feature_scope": f"top_{args.top_numeric}_numeric_by_reference_gain_plus_all_categoricals",
        "objective": "binary_logistic",
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "rounds": int(args.rounds),
        "xgboost_params": params,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "reference_run": str(REFERENCE_CLASSIFIER_RUN),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "reduced_feature_counts": {
            "numeric": len(numeric),
            "categorical": len(categorical),
            "feature_names": len(artifact.feature_names),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote pruned full-label incidence classifier to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
