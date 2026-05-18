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

from train_amount_band_incidence import BAND_COLUMNS, BAND_VALUES, band_labels, class_weights
from train_cumulative_amount_incidence import label_for as cumulative_label_for
from train_incidence_classifier import drop_matching_features, parse_string_list
from train_low_positive_veto import labels as low_veto_labels
from train_stage_model import (
    DEFAULT_DATASET_DIR,
    EncodingArtifact,
    apply_feature_drops,
    artifact_payload,
    best_iteration_end,
    fit_encoder,
    load_dataset,
    transform_frame,
    xgb_params,
)
from train_tail_layer import (
    low_target_labels,
    target_band_labels,
    target_tail_labels,
)


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "JulyScoringFullLabel" / "sip_full_deployment_models"
REFERENCE_AMOUNT_RUN = NEW_DIR / "output" / "model_runs" / "sip" / "regressor_positive" / "quantile_40" / "20260517_124102"
REFERENCE_TWEEDIE_RUN = NEW_DIR / "output" / "model_runs" / "sip" / "regressor_positive" / "tweedie" / "20260517_003424"
REFERENCE_TAIL_RUN = NEW_DIR / "output" / "tail_layer_runs" / "sip" / "regressor_positive" / "tail_ge3p0" / "20260517_185301"
REFERENCE_BASE_INCIDENCE_RUN = NEW_DIR / "output" / "model_runs" / "sip" / "classifier_all" / "binary_logistic" / "20260517_211635"
REFERENCE_AVG_INCIDENCE_RUN = NEW_DIR / "output" / "model_runs" / "sip" / "classifier_all" / "binary_logistic" / "20260517_232319"
BASE_DROP_SUBSTRINGS = "affinity,embedding_pca_,embedding_analog,_avg_sales,_obs"
AVG_DROP_SUBSTRINGS = "affinity,embedding_pca_,embedding_analog,_obs"
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_rounds(model_path: Path) -> int:
    booster = xgb.Booster()
    booster.load_model(model_path)
    best_iteration = getattr(booster, "best_iteration", None)
    return int(best_iteration) + 1 if best_iteration is not None else booster.num_boosted_rounds()


def effective_contract(contract: dict[str, object], drop_substrings: str | None = None) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract, drop_report = apply_feature_drops(contract, "SIP")
    substring_report: dict[str, object] = {"requested_substrings": [], "dropped_numeric_features": [], "dropped_categorical_features": []}
    if drop_substrings:
        contract, substring_report = drop_matching_features(contract, parse_string_list(drop_substrings))
    return contract, drop_report, substring_report


def train_regressor(dataset_dir: Path, output_dir: Path, reference_run: Path, model_name: str) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "SIP", "regressor_positive")
    contract, drop_report, substring_report = effective_contract(contract)
    reference_summary = load_json(reference_run / "run_summary.json")
    objective = str(reference_summary["objective"])
    weighting = str(reference_summary["weighting"])
    params = reference_summary["xgboost_params"]
    rounds = best_rounds(reference_run / "model.json")
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    encoded = transform_frame(frame, artifact, objective, weighting)
    dtrain = xgb.DMatrix(encoded.matrix, label=encoded.labels, weight=encoded.weights, feature_names=artifact.feature_names)
    log(f"Training full SIP {model_name}: rows={len(frame)}, objective={objective}, rounds={rounds}")
    booster = xgb.train(
        params=xgb_params(
            objective,
            seed=42,
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            eta=float(params["eta"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
        ),
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train")],
        verbose_eval=25,
    )
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(run_dir / "model.json")
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "SIP",
        "stage": "regressor_positive",
        "training_scope": "all_labeled_positive_rows",
        "rows": int(len(frame)),
        "objective": objective,
        "weighting": weighting,
        "rounds": int(rounds),
        "xgboost_params": params,
        "reference_run": str(reference_run),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "rounds": int(rounds), "objective": objective}


def train_binary_model(
    frame: pd.DataFrame,
    artifact: EncodingArtifact,
    labels: np.ndarray,
    params: dict[str, object],
    rounds: int,
    output_path: Path,
    name: str,
    scale_pos_weight_multiplier: float = 1.0,
) -> dict[str, object]:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    scale_pos_weight = (negatives / positives if positives > 0 else 1.0) * scale_pos_weight_multiplier
    dtrain = xgb.DMatrix(encoded.matrix, label=labels.astype(np.float32), feature_names=artifact.feature_names)
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
    log(f"Training {name}: rows={len(frame)}, positive_rate={positives / max(len(labels), 1):.5f}, rounds={rounds}")
    booster = xgb.train(train_params, dtrain, num_boost_round=rounds, evals=[(dtrain, "train")], verbose_eval=25)
    booster.save_model(output_path)
    return {
        "rows": int(len(frame)),
        "positive_rate": float(positives / max(len(labels), 1)),
        "rounds": int(rounds),
        "xgboost_params": train_params,
    }


def train_full_tail_layer(dataset_dir: Path, output_dir: Path, amount_run_dir: Path, tweedie_run_dir: Path) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "SIP", "regressor_positive")
    contract, drop_report, substring_report = effective_contract(contract)
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    reference_summary = load_json(REFERENCE_TAIL_RUN / "run_summary.json")
    params = reference_summary["classifier_params"]
    run_dir = output_dir / "tail_layer"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tail_encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")

    tail_rounds = best_rounds(REFERENCE_TAIL_RUN / "tail_classifier.json")
    band_rounds = best_rounds(REFERENCE_TAIL_RUN / "band_classifier.json")
    low_rounds = best_rounds(REFERENCE_TAIL_RUN / "low_classifier.json")
    tail_y = target_tail_labels(frame, float(reference_summary["tail_quantile"]), reference_summary["tail_threshold"])
    band_layer = reference_summary["band_layer"]
    low_layer = reference_summary["low_layer"]
    band_y = target_band_labels(
        frame,
        float(band_layer["lower_quantile"]),
        float(band_layer["upper_quantile"]),
        float(band_layer["target_min"]) if band_layer.get("target_min") is not None else None,
        float(band_layer["target_max"]) if band_layer.get("target_max") is not None else None,
    )
    low_y = low_target_labels(frame, float(low_layer["target_max"]))
    tail_info = train_binary_model(frame, artifact, tail_y, params, tail_rounds, run_dir / "tail_classifier.json", "full SIP tail classifier", float(params.get("scale_pos_weight_multiplier", 1.0)))
    band_info = train_binary_model(frame, artifact, band_y, params, band_rounds, run_dir / "band_classifier.json", "full SIP band classifier", float(params.get("scale_pos_weight_multiplier", 1.0)))
    low_info = train_binary_model(frame, artifact, low_y, params, low_rounds, run_dir / "low_classifier.json", "full SIP low classifier", float(params.get("scale_pos_weight_multiplier", 1.0)))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "SIP",
        "stage": "regressor_positive",
        "training_scope": "all_labeled_positive_rows",
        "base_run_dir": str(amount_run_dir),
        "base_objective": "quantile_40",
        "external_signal_run_dir": str(tweedie_run_dir),
        "tail_quantile": reference_summary["tail_quantile"],
        "tail_threshold": reference_summary["tail_threshold"],
        "tail_target": reference_summary["tail_target"],
        "low_layer": low_layer,
        "band_layer": band_layer,
        "classifier_params": params,
        "multiplier_mode": reference_summary.get("multiplier_mode", "rank"),
        "best_multiplier": {"alpha": 5.4, "band_alpha": 0.0, "external_alpha": 0.4, "low_alpha": 0.6, "scale": 0.345},
        "tail_classifier": tail_info,
        "band_classifier": band_info,
        "low_classifier": low_info,
        "reference_run": str(REFERENCE_TAIL_RUN),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "tail_rounds": tail_rounds, "band_rounds": band_rounds, "low_rounds": low_rounds}


def train_incidence(dataset_dir: Path, output_dir: Path, reference_run: Path, name: str, drop_substrings: str) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "SIP", "classifier_all")
    contract, drop_report, substring_report = effective_contract(contract, drop_substrings)
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    reference_summary = load_json(reference_run / "run_summary.json")
    params = reference_summary["xgboost_params"]
    rounds = best_rounds(reference_run / "model.json")
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    y = frame["positive_sale_flag"].to_numpy(dtype=np.float32)
    info = train_binary_model(frame, artifact, y, params, rounds, f"{run_dir}/model.json", f"full SIP {name}")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "SIP",
        "stage": "classifier_all",
        "training_scope": "all_labeled_rows",
        "objective": "binary_logistic",
        "rounds": int(rounds),
        "reference_run": str(reference_run),
        "xgboost_params": info["xgboost_params"],
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "rounds": int(rounds), "positive_rate": float(y.mean())}


def predict_binary(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    return booster.predict(dmatrix, iteration_range=(0, best_iteration_end(booster))).astype(np.float32)


def load_artifact(path: Path) -> EncodingArtifact:
    from train_tail_layer import artifact_from_payload

    return artifact_from_payload(load_json(path))


def train_amount_band(dataset_dir: Path, output_dir: Path, reference_run: Path) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "SIP", "classifier_all")
    contract, drop_report, substring_report = effective_contract(contract, AVG_DROP_SUBSTRINGS)
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    reference_summary = load_json(reference_run / "run_summary.json")
    params = reference_summary["xgboost_params"]
    rounds = best_rounds(reference_run / "model.json")
    y = band_labels(frame)
    weights = class_weights(y, float(reference_summary["class_weight_power"]), float(reference_summary["max_class_weight"]))
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dtrain = xgb.DMatrix(encoded.matrix, label=y, weight=weights, feature_names=artifact.feature_names)
    train_params = {
        "objective": "multi:softprob",
        "num_class": len(BAND_COLUMNS),
        "eval_metric": ["mlogloss"],
        "tree_method": "hist",
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "eta": float(params["eta"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "seed": 42,
    }
    run_dir = output_dir / "amount_band_multiclass"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Training full SIP amount-band classifier: rows={len(frame)}, rounds={rounds}")
    booster = xgb.train(train_params, dtrain, num_boost_round=rounds, evals=[(dtrain, "train")], verbose_eval=25)
    booster.save_model(run_dir / "model.json")
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "SIP",
        "stage": "classifier_all",
        "training_scope": "all_labeled_rows",
        "objective": "amount_band_multiclass",
        "rounds": int(rounds),
        "xgboost_params": train_params,
        "class_weight_power": reference_summary["class_weight_power"],
        "max_class_weight": reference_summary["max_class_weight"],
        "band_columns": BAND_COLUMNS,
        "band_values": BAND_VALUES.tolist(),
        "reference_run": str(reference_run),
        "rows": int(len(frame)),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "rounds": int(rounds)}


def train_cumulative(dataset_dir: Path, output_dir: Path, reference_run: Path) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "SIP", "classifier_all")
    contract, drop_report, substring_report = effective_contract(contract, AVG_DROP_SUBSTRINGS)
    artifact = fit_encoder(frame, contract["numeric_features"], contract["categorical_features"])
    reference_summary = load_json(reference_run / "run_summary.json")
    run_dir = output_dir / "cumulative_amount_incidence"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    model_infos = []
    for model in reference_summary["models"]:
        threshold = float(model["threshold"])
        name = str(model["name"])
        params = model["xgboost_params"]
        rounds = best_rounds(reference_run / f"model_{name}.json")
        y = cumulative_label_for(frame, threshold)
        dtrain = xgb.DMatrix(encoded.matrix, label=y, feature_names=artifact.feature_names)
        train_params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr", "auc"],
            "tree_method": "hist",
            "max_depth": int(params["max_depth"]),
            "min_child_weight": float(params["min_child_weight"]),
            "eta": float(params["eta"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "scale_pos_weight": float(params["scale_pos_weight"]),
            "seed": 42,
        }
        log(f"Training full SIP cumulative {name}: rows={len(frame)}, rounds={rounds}")
        booster = xgb.train(train_params, dtrain, num_boost_round=rounds, evals=[(dtrain, "train")], verbose_eval=25)
        booster.save_model(run_dir / f"model_{name}.json")
        model_infos.append({"threshold": threshold, "name": name, "rounds": int(rounds), "xgboost_params": train_params})
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "SIP",
        "stage": "classifier_all",
        "training_scope": "all_labeled_rows",
        "objective": "cumulative_amount_incidence",
        "thresholds": reference_summary["thresholds"],
        "models": model_infos,
        "reference_run": str(reference_run),
        "rows": int(len(frame)),
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(frame)), "models": model_infos}


def train_veto(dataset_dir: Path, output_dir: Path, incidence_run_dir: Path, reference_run: Path, name: str, label_mode: str) -> dict[str, object]:
    frame, manifest, contract = load_dataset(dataset_dir, "SIP", "classifier_all")
    contract, drop_report, substring_report = effective_contract(contract, AVG_DROP_SUBSTRINGS)
    incidence_artifact = load_artifact(incidence_run_dir / "encoding_artifact.json")
    incidence_booster = xgb.Booster()
    incidence_booster.load_model(incidence_run_dir / "model.json")
    frame = frame.copy()
    log(f"Scoring full SIP base raw probability before {name}")
    frame["raw_probability"] = predict_binary(incidence_booster, frame, incidence_artifact)
    candidate_threshold = float(load_json(reference_run / "run_summary.json")["candidate_threshold"])
    candidate = frame.loc[frame["raw_probability"] >= candidate_threshold].copy()
    effective = dict(contract)
    effective["numeric_features"] = list(effective["numeric_features"]) + ["raw_probability"]
    artifact = fit_encoder(candidate, effective["numeric_features"], effective["categorical_features"])
    if label_mode == "zero":
        y = (candidate["sales_target"].to_numpy(dtype=np.float32) <= 0.0).astype(np.float32)
    else:
        y = low_veto_labels(candidate, label_mode)
    ref = load_json(reference_run / "run_summary.json")
    params = ref["xgboost_params"]
    rounds = best_rounds(reference_run / "model.json")
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "encoding_artifact.json").write_text(json.dumps(artifact_payload(artifact), indent=2), encoding="utf-8")
    info = train_binary_model(candidate, artifact, y, params, rounds, run_dir / "model.json", f"full SIP {name}")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": "SIP",
        "stage": "classifier_all",
        "training_scope": "all_labeled_candidate_rows",
        "objective": name,
        "candidate_threshold": candidate_threshold,
        "label_mode": label_mode,
        "reference_run": str(reference_run),
        "incidence_run_dir": str(incidence_run_dir),
        "rows": int(len(candidate)),
        "all_rows": int(len(frame)),
        "xgboost_params": info["xgboost_params"],
        "manifest": manifest,
        "feature_drop_report": drop_report,
        "feature_substring_drop_report": substring_report,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "rows": int(len(candidate)), "rounds": int(rounds)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SIP deployment artifacts on all currently labeled rows.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    amount = train_regressor(args.dataset_dir, model_dir, REFERENCE_AMOUNT_RUN, "regressor_positive_q40")
    tweedie = train_regressor(args.dataset_dir, model_dir, REFERENCE_TWEEDIE_RUN, "regressor_positive_tweedie")
    tail = train_full_tail_layer(args.dataset_dir, model_dir, Path(amount["run_dir"]), Path(tweedie["run_dir"]))
    base_incidence = train_incidence(args.dataset_dir, model_dir, REFERENCE_BASE_INCIDENCE_RUN, "incidence_base", BASE_DROP_SUBSTRINGS)
    avg_incidence = train_incidence(args.dataset_dir, model_dir, REFERENCE_AVG_INCIDENCE_RUN, "incidence_avg_sales", AVG_DROP_SUBSTRINGS)
    amount_band = train_amount_band(args.dataset_dir, model_dir, NEW_DIR / "output" / "model_runs" / "sip" / "classifier_all" / "amount_band_multiclass" / "20260518_001723")
    cumulative = train_cumulative(args.dataset_dir, model_dir, NEW_DIR / "output" / "model_runs" / "sip" / "classifier_all" / "cumulative_amount_incidence" / "20260518_014236")
    zero_veto = train_veto(args.dataset_dir, model_dir, Path(base_incidence["run_dir"]), NEW_DIR / "output" / "model_runs" / "sip" / "classifier_all" / "zero_leak_veto" / "20260518_002153", "zero_leak_veto", "zero")
    low_veto = train_veto(args.dataset_dir, model_dir, Path(base_incidence["run_dir"]), NEW_DIR / "output" / "model_runs" / "sip" / "classifier_all" / "low_positive_veto" / "20260518_021510", "low_positive_veto", "le1")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "training_scope": "all currently labeled SIP rows",
        "models": {
            "amount": amount,
            "tweedie": tweedie,
            "tail": tail,
            "base_incidence": base_incidence,
            "avg_incidence": avg_incidence,
            "amount_band": amount_band,
            "cumulative": cumulative,
            "zero_veto": zero_veto,
            "low_veto": low_veto,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote full-label SIP deployment models to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
