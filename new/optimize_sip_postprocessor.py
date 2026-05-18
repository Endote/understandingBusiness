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

from evaluate_incidence_pipeline import (
    KEY_COLUMNS,
    META_COLUMNS,
    apply_tail_multiplier,
    load_base_regressor,
    load_optional_band_model,
    load_optional_low_model,
    load_tail_model,
    predict_tail_probability,
    read_incidence_predictions,
    read_model_frame,
    tail_multiplier_from_summary,
)
from train_stage_model import DEFAULT_DATASET_DIR, predict_frame
from train_tail_layer import probability_rank


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "pipeline_runs"
BAND_VALUE_COLUMNS = ["band_p0_zero", "band_p1_one", "band_p2_two", "band_p3_three_four", "band_p4_five_plus"]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def rank(values: np.ndarray) -> np.ndarray:
    return probability_rank(values.astype(np.float32)).astype(np.float64)


def read_signal_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=list(dict.fromkeys([*KEY_COLUMNS, *columns])), parse_dates=["onsaledate"])
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path} has duplicate key rows")
    return frame


def fast_metrics(y_true: np.ndarray, prediction: np.ndarray, zero_mask: np.ndarray, positive_decile: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction.astype(np.float64), 0.0, None)
    actual_sum = float(y_true.sum())
    predicted_sum = float(prediction.sum())
    zero_predicted_sum = float(prediction[zero_mask].sum())
    positive_mask = y_true > 0.0
    decile_pred = np.bincount(positive_decile[positive_mask], weights=prediction[positive_mask], minlength=10)
    decile_actual = np.bincount(positive_decile[positive_mask], weights=y_true[positive_mask], minlength=10)
    ratios = decile_pred / np.maximum(decile_actual, 1e-9)
    wape = float(np.abs(y_true - prediction).sum() / max(actual_sum, 1e-9))
    return {
        "wape": wape,
        "total": safe_ratio(predicted_sum, actual_sum),
        "zero": safe_ratio(zero_predicted_sum, actual_sum),
        "bottom30": float(ratios[:3].mean()),
        "top30": float(ratios[7:10].mean()),
        "top10": float(ratios[9]),
        "max_decile": float(ratios.max()),
        "min_decile": float(ratios.min()),
        "d1": float(ratios[0]),
        "d2": float(ratios[1]),
        "d3": float(ratios[2]),
        "d8": float(ratios[7]),
        "d9": float(ratios[8]),
        "d10": float(ratios[9]),
        "predicted_sum": predicted_sum,
        "actual_sum": actual_sum,
    }


def sip_objective(metrics: dict[str, float]) -> float:
    return float(
        22.0 * max(metrics["zero"] - 0.75, 0.0)
        + 12.0 * max(metrics["total"] - 1.45, 0.0)
        + 3.0 * max(0.95 - metrics["total"], 0.0)
        + 10.0 * max(metrics["max_decile"] - 1.40, 0.0)
        + 7.0 * max(metrics["bottom30"] - 1.30, 0.0)
        + 9.0 * max(0.72 - metrics["top30"], 0.0)
        + 8.0 * max(0.82 - metrics["top10"], 0.0)
        + 3.0 * max(0.58 - metrics["min_decile"], 0.0)
        + 0.20 * metrics["wape"]
    )


def positive_deciles(y_true: np.ndarray) -> np.ndarray:
    result = np.full(len(y_true), -1, dtype=np.int32)
    positive_idx = np.flatnonzero(y_true > 0.0)
    order = np.argsort(np.arange(len(positive_idx)), kind="stable")
    # Match the existing qcut(rank(method="first")) behavior: equal-sized bins over positive row order by sales rank.
    sales = y_true[positive_idx]
    rank_order = np.lexsort((np.arange(len(sales)), sales))
    deciles = np.floor(np.arange(len(sales), dtype=np.float64) * 10.0 / len(sales)).astype(np.int32)
    deciles = np.minimum(deciles, 9)
    assigned = np.empty(len(sales), dtype=np.int32)
    assigned[rank_order] = deciles
    result[positive_idx] = assigned[order]
    return result


def build_signal_frame(args: argparse.Namespace, split: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    base_booster, base_artifact, base_objective = load_base_regressor(args.base_run_dir)
    tail_booster, tail_artifact, tail_summary = load_tail_model(args.tail_run_dir)
    band_booster = load_optional_band_model(args.tail_run_dir)
    low_booster = load_optional_low_model(args.tail_run_dir)
    external_run_dir = args.external_signal_run_dir
    if external_run_dir is None and tail_summary.get("external_signal_run_dir"):
        external_run_dir = Path(str(tail_summary["external_signal_run_dir"]))
    external_booster = external_artifact = external_objective = None
    if external_run_dir is not None:
        external_booster, external_artifact, external_objective = load_base_regressor(external_run_dir)

    needed_columns = [
        *META_COLUMNS,
        *base_artifact.numeric_features,
        *base_artifact.categorical_features,
        *tail_artifact.numeric_features,
        *tail_artifact.categorical_features,
    ]
    if external_artifact is not None:
        needed_columns.extend([*external_artifact.numeric_features, *external_artifact.categorical_features])
    frame = read_model_frame(args.dataset_dir, args.family, needed_columns)
    frame = frame.loc[frame["split"] == split].copy()

    base_incidence = read_incidence_predictions(args.base_calibration_run_dir / f"{'validation' if split == 'valid' else split}_calibrated_predictions.csv", split)
    avg_incidence = read_incidence_predictions(args.avg_calibration_run_dir / f"{'validation' if split == 'valid' else split}_calibrated_predictions.csv", split)
    base_incidence = base_incidence.rename(
        columns={"raw_probability": "base_raw_probability", "calibrated_probability": "base_calibrated_probability"}
    )
    avg_incidence = avg_incidence.rename(
        columns={"raw_probability": "avg_raw_probability", "calibrated_probability": "avg_calibrated_probability"}
    )
    frame = frame.merge(base_incidence, on=KEY_COLUMNS, how="left", validate="one_to_one")
    frame = frame.merge(avg_incidence, on=KEY_COLUMNS, how="left", validate="one_to_one")

    if args.amount_band_run_dir is not None:
        band_file = args.amount_band_run_dir / f"{'validation' if split == 'valid' else split}_amount_band_predictions.csv"
        band = read_signal_csv(band_file, ["band_positive_probability", "band_tail_probability", "band_expected_units"])
        frame = frame.merge(band, on=KEY_COLUMNS, how="left", validate="one_to_one")
    else:
        frame["band_positive_probability"] = frame["base_raw_probability"]
        frame["band_tail_probability"] = frame["base_raw_probability"]
        frame["band_expected_units"] = frame["base_raw_probability"]

    if args.veto_run_dir is not None:
        veto_file = args.veto_run_dir / f"{'validation' if split == 'valid' else split}_veto_predictions.csv"
        veto = read_signal_csv(veto_file, ["veto_zero_probability"])
        frame = frame.merge(veto, on=KEY_COLUMNS, how="left", validate="one_to_one")
    else:
        frame["veto_zero_probability"] = 0.0

    multiplier_override = {
        "alpha": args.tail_alpha,
        "band_alpha": args.tail_band_alpha,
        "external_alpha": args.tail_external_alpha,
        "low_alpha": args.tail_low_alpha,
        "scale": args.tail_scale,
    }
    log(f"Predicting amount/tail signals for {split}: {len(frame)} rows")
    base_prediction = predict_frame(base_booster, frame, base_artifact, base_objective)
    tail_probability = predict_tail_probability(tail_booster, frame, tail_artifact)
    band_probability = predict_tail_probability(band_booster, frame, tail_artifact) if band_booster is not None else None
    low_probability = predict_tail_probability(low_booster, frame, tail_artifact) if low_booster is not None else None
    external_signal = (
        predict_frame(external_booster, frame, external_artifact, external_objective)
        if external_booster is not None and external_artifact is not None and external_objective is not None
        else None
    )
    regressor_prediction = apply_tail_multiplier(
        base_prediction,
        tail_probability,
        tail_summary,
        band_probability=band_probability,
        low_probability=low_probability,
        external_signal=external_signal,
        multiplier_override=multiplier_override,
    )

    signals = {
        "regressor": regressor_prediction.astype(np.float64),
        "base_raw": frame["base_raw_probability"].to_numpy(dtype=np.float64),
        "base_calibrated": frame["base_calibrated_probability"].to_numpy(dtype=np.float64),
        "avg_raw": frame["avg_raw_probability"].to_numpy(dtype=np.float64),
        "avg_calibrated": frame["avg_calibrated_probability"].to_numpy(dtype=np.float64),
        "band_positive": frame["band_positive_probability"].fillna(0.0).to_numpy(dtype=np.float64),
        "band_tail": frame["band_tail_probability"].fillna(0.0).to_numpy(dtype=np.float64),
        "band_expected": frame["band_expected_units"].fillna(0.0).to_numpy(dtype=np.float64),
        "veto": frame["veto_zero_probability"].fillna(0.0).to_numpy(dtype=np.float64),
        "tail": tail_probability.astype(np.float64),
        "low": low_probability.astype(np.float64) if low_probability is not None else np.zeros(len(frame), dtype=np.float64),
    }
    summary_payload = {
        "external_signal_run_dir": str(external_run_dir) if external_run_dir is not None else None,
        "tail_multiplier": tail_multiplier_from_summary(tail_summary, multiplier_override),
    }
    return frame, signals, summary_payload


def generate_candidates(rng: np.random.Generator, n_trials: int) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    anchors = [
        {"base": 1.0, "avg": 0.0, "band": 0.0, "tail": 0.0, "veto": 0.0, "threshold": 0.84, "floor": 0.0, "scale": 1.0},
        {"base": 1.0, "avg": 0.0, "band": 0.0, "tail": 0.0, "veto": 0.0, "threshold": 0.84, "floor": 0.025, "scale": 1.0},
        {"base": 1.0, "avg": 0.0, "band": 0.0, "tail": 0.0, "veto": 0.0, "threshold": 0.88, "floor": 0.05, "scale": 1.0},
        {"base": 0.7, "avg": 0.2, "band": 0.1, "tail": 0.0, "veto": 0.0, "threshold": 0.86, "floor": 0.025, "scale": 1.0},
        {"base": 0.55, "avg": 0.25, "band": 0.15, "tail": 0.05, "veto": 0.10, "threshold": 0.86, "floor": 0.025, "scale": 0.98},
    ]
    candidates.extend(anchors)
    for _ in range(n_trials):
        raw = rng.dirichlet(np.array([3.2, 1.7, 1.4, 0.8], dtype=np.float64))
        candidates.append(
            {
                "base": float(raw[0]),
                "avg": float(raw[1]),
                "band": float(raw[2]),
                "tail": float(raw[3]),
                "veto": float(rng.uniform(0.0, 0.45)),
                "threshold": float(rng.uniform(0.82, 0.96)),
                "floor": float(rng.choice([0.0, 0.01, 0.025, 0.04, 0.05, 0.075])),
                "scale": float(rng.uniform(0.90, 1.06)),
            }
        )
    return candidates


def predict_policy(signals: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    base_rank = signals.get("base_raw_rank")
    avg_rank = signals.get("avg_raw_rank")
    band_rank = signals.get("band_expected_rank")
    tail_rank = signals.get("tail_rank")
    veto_rank = signals.get("veto_rank")
    if base_rank is None or avg_rank is None or band_rank is None or tail_rank is None or veto_rank is None:
        base_rank = rank(signals["base_raw"])
        avg_rank = rank(signals["avg_raw"])
        band_rank = rank(signals["band_expected"])
        tail_rank = rank(signals["tail"])
        veto_rank = rank(signals["veto"])
    combined = (
        params["base"] * base_rank
        + params["avg"] * avg_rank
        + params["band"] * band_rank
        + params["tail"] * tail_rank
        - params["veto"] * veto_rank
    )
    cutoff = float(np.quantile(combined, params["threshold"]))
    gate = (combined >= cutoff).astype(np.float64)
    layer = params["floor"] + (1.0 - params["floor"]) * gate
    return signals["regressor"] * layer * params["scale"]


def add_precomputed_ranks(signals: dict[str, np.ndarray]) -> None:
    for key in ("base_raw", "avg_raw", "band_expected", "tail", "veto"):
        signals[f"{key}_rank"] = rank(signals[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize SIP postprocessor over incidence, band, veto, and tail signals.")
    parser.add_argument("--family", choices=["SIP"], default="SIP")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-calibration-run-dir", type=Path, required=True)
    parser.add_argument("--avg-calibration-run-dir", type=Path, required=True)
    parser.add_argument("--amount-band-run-dir", type=Path, default=None)
    parser.add_argument("--veto-run-dir", type=Path, default=None)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--tail-run-dir", type=Path, required=True)
    parser.add_argument("--external-signal-run-dir", type=Path, default=None)
    parser.add_argument("--tail-alpha", type=float, default=None)
    parser.add_argument("--tail-band-alpha", type=float, default=None)
    parser.add_argument("--tail-external-alpha", type=float, default=None)
    parser.add_argument("--tail-low-alpha", type=float, default=None)
    parser.add_argument("--tail-scale", type=float, default=None)
    parser.add_argument("--trials", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--top-k-test", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = args.output_dir / args.family.lower() / "postprocessor_optimization" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    valid_frame, valid_signals, signal_summary = build_signal_frame(args, "valid")
    test_frame, test_signals, _ = build_signal_frame(args, "test")
    add_precomputed_ranks(valid_signals)
    add_precomputed_ranks(test_signals)
    valid_y = valid_frame["sales_target"].to_numpy(dtype=np.float64)
    test_y = test_frame["sales_target"].to_numpy(dtype=np.float64)
    valid_zero = valid_y <= 0.0
    test_zero = test_y <= 0.0
    valid_decile = positive_deciles(valid_y)
    test_decile = positive_deciles(test_y)

    candidates = generate_candidates(rng, args.trials)
    rows = []
    log(f"Optimizing {len(candidates)} postprocessor candidates on validation")
    for idx, params in enumerate(candidates):
        prediction = predict_policy(valid_signals, params)
        metrics = fast_metrics(valid_y, prediction, valid_zero, valid_decile)
        metrics["sip_objective"] = sip_objective(metrics)
        metrics["candidate_id"] = idx
        metrics.update(params)
        rows.append(metrics)
    validation_summary = pd.DataFrame.from_records(rows).sort_values("sip_objective")
    validation_summary.to_csv(run_dir / "validation_postprocessor_candidates.csv", index=False)

    test_rows = []
    selected_ids = validation_summary.head(args.top_k_test)["candidate_id"].astype(int).tolist()
    for candidate_id in selected_ids:
        params = candidates[candidate_id]
        prediction = predict_policy(test_signals, params)
        metrics = fast_metrics(test_y, prediction, test_zero, test_decile)
        metrics["sip_objective"] = sip_objective(metrics)
        metrics["candidate_id"] = candidate_id
        metrics.update(params)
        test_rows.append(metrics)
    test_summary = pd.DataFrame.from_records(test_rows).sort_values("sip_objective")
    test_summary.to_csv(run_dir / "test_postprocessor_candidates.csv", index=False)

    selected = validation_summary.iloc[0].to_dict()
    selected_test = test_summary.loc[test_summary["candidate_id"] == int(selected["candidate_id"])].iloc[0].to_dict()
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "base_calibration_run_dir": str(args.base_calibration_run_dir),
        "avg_calibration_run_dir": str(args.avg_calibration_run_dir),
        "amount_band_run_dir": str(args.amount_band_run_dir) if args.amount_band_run_dir else None,
        "veto_run_dir": str(args.veto_run_dir) if args.veto_run_dir else None,
        "base_run_dir": str(args.base_run_dir),
        "tail_run_dir": str(args.tail_run_dir),
        "signal_summary": signal_summary,
        "trials": args.trials,
        "selected_validation": selected,
        "selected_test": selected_test,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Wrote postprocessor optimization outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
