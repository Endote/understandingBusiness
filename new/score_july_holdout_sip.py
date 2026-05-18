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

from evaluate_incidence_pipeline import apply_tail_multiplier, load_optional_band_model, load_optional_low_model, load_tail_model, predict_tail_probability
from score_july_holdout_weeklies import (
    DEFAULT_DB_URI,
    DEFAULT_DATASET_DIR,
    META_COLUMNS,
    add_completed_prior_features,
    add_derived_numeric_features,
    add_stage_embedding_features,
    grouped_summary,
    load_family_embeddings,
    load_holdout_embeddings,
    load_holdout_frame,
    prediction_summary,
)
from train_amount_band_incidence import BAND_VALUES, normalize_probability
from train_stage_model import EncodingArtifact, predict_frame, transform_frame
from train_tail_layer import artifact_from_payload, load_base_regressor, probability_rank


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "JulyScoringFullLabel" / "sip"
KEY_COLUMNS = ["store_id", "product_id", "onsaledate"]
BASE_892 = {
    "base": 0.529573,
    "avg": 0.056413,
    "band": 0.367175,
    "tail": 0.046838,
    "veto": 0.012264,
    "threshold": 0.917195,
    "scale": 1.018110,
}
LIFT_11655 = {
    "ge3": 0.360329,
    "ge4": 0.439266,
    "ge5": 0.057644,
    "band": 0.022452,
    "tail": 0.120309,
    "low": 0.806411,
    "zero_weight": 0.193589,
    "boost_quantile": 0.741205,
    "boost": 0.357234,
    "damp_quantile": 0.720835,
    "damp": 0.374016,
    "scale": 0.973659,
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact(path: Path) -> EncodingArtifact:
    return artifact_from_payload(load_json(path))


def predict_probability(booster: xgb.Booster, frame: pd.DataFrame, artifact: EncodingArtifact) -> np.ndarray:
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    return booster.predict(dmatrix).astype(np.float32)


def rank(values: np.ndarray) -> np.ndarray:
    return probability_rank(values.astype(np.float32)).astype(np.float64)


def score_binary_model(run_dir: Path, frame: pd.DataFrame, model_name: str = "model.json") -> np.ndarray:
    artifact = load_artifact(run_dir / "encoding_artifact.json")
    booster = xgb.Booster()
    booster.load_model(run_dir / model_name)
    return predict_probability(booster, frame, artifact)


def score_amount_band(run_dir: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact = load_artifact(run_dir / "encoding_artifact.json")
    booster = xgb.Booster()
    booster.load_model(run_dir / "model.json")
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    probability = normalize_probability(booster.predict(dmatrix))
    positive_probability = 1.0 - probability[:, 0]
    tail_probability = probability[:, 3] + probability[:, 4]
    expected_units = probability @ BAND_VALUES
    return positive_probability.astype(np.float32), tail_probability.astype(np.float32), expected_units.astype(np.float32)


def score_cumulative(run_dir: Path, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    artifact = load_artifact(run_dir / "encoding_artifact.json")
    encoded = transform_frame(frame, artifact, "tweedie", "uniform")
    dmatrix = xgb.DMatrix(encoded.matrix, feature_names=artifact.feature_names)
    result: dict[str, np.ndarray] = {}
    for threshold in (1, 2, 3, 4, 5):
        booster = xgb.Booster()
        booster.load_model(run_dir / f"model_ge{threshold}.json")
        result[f"cum_ge{threshold}"] = booster.predict(dmatrix).astype(np.float32)
    return result


def apply_892(
    positive_prediction: np.ndarray,
    base_raw: np.ndarray,
    avg_raw: np.ndarray,
    band_expected: np.ndarray,
    tail_probability: np.ndarray,
    zero_veto: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = (
        BASE_892["base"] * rank(base_raw)
        + BASE_892["avg"] * rank(avg_raw)
        + BASE_892["band"] * rank(band_expected)
        + BASE_892["tail"] * rank(tail_probability)
        - BASE_892["veto"] * rank(zero_veto)
    )
    combined_rank = rank(combined)
    gate = (combined_rank >= BASE_892["threshold"]).astype(np.float32)
    prediction = np.clip(positive_prediction.astype(np.float64) * gate * BASE_892["scale"], 0.0, None).astype(np.float32)
    return prediction, gate, combined_rank.astype(np.float32)


def apply_lift_11655(
    base_prediction: np.ndarray,
    gate: np.ndarray,
    cumulative: dict[str, np.ndarray],
    band_tail: np.ndarray,
    tail_probability: np.ndarray,
    low_veto: np.ndarray,
    zero_veto: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    boost_score = (
        LIFT_11655["ge3"] * rank(cumulative["cum_ge3"])
        + LIFT_11655["ge4"] * rank(cumulative["cum_ge4"])
        + LIFT_11655["ge5"] * rank(cumulative["cum_ge5"])
        + LIFT_11655["band"] * rank(band_tail)
        + LIFT_11655["tail"] * rank(tail_probability)
    )
    damp_score = LIFT_11655["low"] * rank(low_veto) + LIFT_11655["zero_weight"] * rank(zero_veto)
    active = gate > 0
    boost_cutoff = float(np.quantile(boost_score[active], LIFT_11655["boost_quantile"])) if np.any(active) else math.inf
    damp_cutoff = float(np.quantile(damp_score[active], LIFT_11655["damp_quantile"])) if np.any(active) else math.inf
    boost_gate = (active & (boost_score >= boost_cutoff)).astype(np.float32)
    damp_gate = (active & (damp_score >= damp_cutoff)).astype(np.float32)
    multiplier = (1.0 + LIFT_11655["boost"] * boost_gate) * (1.0 - LIFT_11655["damp"] * damp_gate) * LIFT_11655["scale"]
    prediction = np.clip(base_prediction.astype(np.float64) * multiplier, 0.0, None).astype(np.float32)
    return prediction, multiplier.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SIP July holdout rows with full-label SIP deployment models.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    family = "SIP"
    family_key = family.lower()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_run_dir / "models"

    log("Loading SIP manifests")
    amount_manifest = load_json(args.dataset_dir / family_key / "manifest_regressor_positive.json")
    classifier_manifest = load_json(args.dataset_dir / family_key / "manifest_classifier_all.json")
    frame = load_holdout_frame(args.db_uri, family)
    if frame.empty:
        raise ValueError("No SIP holdout rows found to score")
    duplicate_rows = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_rows:
        raise ValueError(f"Holdout has {duplicate_rows} duplicate store/product/onsale keys")

    log("Adding completed-history priors from core")
    base_frame, prior_summary = add_completed_prior_features(frame, family, args.db_uri)
    core_products, core_matrix = load_family_embeddings(args.db_uri, family)
    holdout_products, holdout_matrix, missing_embeddings = load_holdout_embeddings(args.db_uri, family)

    log("Building amount-stage holdout features")
    amount_features, amount_embedding_summary = add_stage_embedding_features(
        base_frame,
        args.db_uri,
        family,
        args.dataset_dir,
        "regressor_positive",
        core_products,
        core_matrix,
        holdout_products,
        holdout_matrix,
        amount_manifest,
    )
    amount_features = add_derived_numeric_features(amount_features)

    log("Building incidence-stage holdout features")
    incidence_features, incidence_embedding_summary = add_stage_embedding_features(
        base_frame,
        args.db_uri,
        family,
        args.dataset_dir,
        "classifier_all",
        core_products,
        core_matrix,
        holdout_products,
        holdout_matrix,
        classifier_manifest,
    )
    incidence_features = add_derived_numeric_features(incidence_features)

    log("Scoring positive amount stack")
    base_booster, base_artifact, base_objective = load_base_regressor(model_dir / "regressor_positive_q40")
    q40_prediction = predict_frame(base_booster, amount_features, base_artifact, base_objective).astype(np.float32)
    tweedie_booster, tweedie_artifact, tweedie_objective = load_base_regressor(model_dir / "regressor_positive_tweedie")
    tweedie_signal = predict_frame(tweedie_booster, amount_features, tweedie_artifact, tweedie_objective).astype(np.float32)
    tail_booster, tail_artifact, tail_summary = load_tail_model(model_dir / "tail_layer")
    tail_probability = predict_tail_probability(tail_booster, amount_features, tail_artifact).astype(np.float32)
    band_booster = load_optional_band_model(model_dir / "tail_layer")
    low_booster = load_optional_low_model(model_dir / "tail_layer")
    tail_band_probability = predict_tail_probability(band_booster, amount_features, tail_artifact).astype(np.float32)
    tail_low_probability = predict_tail_probability(low_booster, amount_features, tail_artifact).astype(np.float32)
    positive_prediction = apply_tail_multiplier(
        q40_prediction,
        tail_probability,
        tail_summary,
        band_probability=tail_band_probability,
        low_probability=tail_low_probability,
        external_signal=tweedie_signal,
        multiplier_override={"alpha": 5.4, "band_alpha": 0.0, "external_alpha": 0.4, "low_alpha": 0.6, "scale": 0.345},
    ).astype(np.float32)

    log("Scoring incidence and auxiliary classifiers")
    incidence_base = score_binary_model(model_dir / "incidence_base", incidence_features)
    incidence_avg = score_binary_model(model_dir / "incidence_avg_sales", incidence_features)
    band_positive, band_tail, band_expected = score_amount_band(model_dir / "amount_band_multiclass", incidence_features)
    cumulative = score_cumulative(model_dir / "cumulative_amount_incidence", incidence_features)

    zero_veto_features = incidence_features.copy()
    zero_veto_features["raw_probability"] = incidence_base
    zero_veto = score_binary_model(model_dir / "zero_leak_veto", zero_veto_features)
    low_veto_features = incidence_features.copy()
    low_veto_features["raw_probability"] = incidence_base
    low_veto = score_binary_model(model_dir / "low_positive_veto", low_veto_features)

    prediction_892, gate_892, rank_892 = apply_892(
        positive_prediction,
        incidence_base,
        incidence_avg,
        band_expected,
        tail_probability,
        zero_veto,
    )
    prediction_lift, lift_multiplier = apply_lift_11655(
        prediction_892,
        gate_892,
        cumulative,
        band_tail,
        tail_probability,
        low_veto,
        zero_veto,
    )

    output = frame[META_COLUMNS].copy()
    output["incidence_base_raw_probability"] = incidence_base
    output["incidence_avg_raw_probability"] = incidence_avg
    output["incidence_raw_probability"] = incidence_base
    output["incidence_calibrated_probability"] = incidence_avg
    output["band_positive_probability"] = band_positive
    output["band_tail_probability"] = band_tail
    output["band_expected_units"] = band_expected
    for key, value in cumulative.items():
        output[f"{key}_probability"] = value
    output["zero_veto_probability"] = zero_veto
    output["low_positive_veto_probability"] = low_veto
    output["positive_q40_prediction"] = q40_prediction
    output["positive_tweedie_signal"] = tweedie_signal
    output["tail_probability"] = tail_probability
    output["tail_band_probability"] = tail_band_probability
    output["tail_low_probability"] = tail_low_probability
    output["positive_regressor_prediction"] = positive_prediction
    output["sip_892_rank"] = rank_892
    output["sip_892_gate"] = gate_892
    output["sip_prediction_892"] = prediction_892
    output["sip_lift_11655_multiplier"] = lift_multiplier
    output["sip_prediction_lift_11655"] = prediction_lift
    output["july_holdout_prediction"] = prediction_892

    predictions_csv = run_dir / "july_holdout_sip_predictions.csv"
    predictions_parquet = run_dir / "july_holdout_sip_predictions.parquet"
    output.to_csv(predictions_csv, index=False)
    output.to_parquet(predictions_parquet, index=False)
    prediction_summary(output, ["positive_regressor_prediction", "sip_prediction_892", "sip_prediction_lift_11655", "july_holdout_prediction"]).to_csv(
        run_dir / "prediction_summary.csv",
        index=False,
    )
    grouped_summary(output, ["onsaledate"], "july_holdout_prediction").to_csv(run_dir / "summary_by_onsaledate.csv", index=False)
    grouped_summary(output, ["title"], "july_holdout_prediction").to_csv(run_dir / "summary_by_title.csv", index=False)
    grouped_summary(output, ["classoftrade"], "july_holdout_prediction").to_csv(run_dir / "summary_by_classoftrade.csv", index=False)
    grouped_summary(output, ["store_chain"], "july_holdout_prediction").to_csv(run_dir / "summary_by_store_chain.csv", index=False)
    if not missing_embeddings.empty:
        missing_embeddings.to_csv(run_dir / "missing_holdout_embeddings.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": family,
        "run_dir": str(run_dir),
        "model_run_dir": str(args.model_run_dir),
        "row_count": int(len(output)),
        "product_count": int(output["product_id"].nunique()),
        "store_count": int(output["store_id"].nunique()),
        "onsaledate_min": output["onsaledate"].min().strftime("%Y-%m-%d"),
        "onsaledate_max": output["onsaledate"].max().strftime("%Y-%m-%d"),
        "main_prediction": "sip_prediction_892",
        "research_prediction": "sip_prediction_lift_11655",
        "base_892": BASE_892,
        "lift_11655": LIFT_11655,
        "prior_features": prior_summary,
        "amount_embedding_features": amount_embedding_summary,
        "incidence_embedding_features": incidence_embedding_summary,
        "holdout_embedding_missing_products": missing_embeddings.to_dict(orient="records"),
        "outputs": {
            "predictions_csv": str(predictions_csv),
            "predictions_parquet": str(predictions_parquet),
            "prediction_summary": str(run_dir / "prediction_summary.csv"),
            "summary_by_onsaledate": str(run_dir / "summary_by_onsaledate.csv"),
            "summary_by_title": str(run_dir / "summary_by_title.csv"),
            "summary_by_classoftrade": str(run_dir / "summary_by_classoftrade.csv"),
            "summary_by_store_chain": str(run_dir / "summary_by_store_chain.csv"),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"Wrote SIP July holdout scoring outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
