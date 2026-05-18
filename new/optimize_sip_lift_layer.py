#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_sip_postprocessor import (
    DEFAULT_OUTPUT_DIR,
    KEY_COLUMNS,
    build_signal_frame,
    fast_metrics,
    positive_deciles,
    rank,
    read_signal_csv,
)
from train_stage_model import DEFAULT_DATASET_DIR


BASE_892 = {
    "base": 0.529573,
    "avg": 0.056413,
    "band": 0.367175,
    "tail": 0.046838,
    "veto": 0.012264,
    "threshold": 0.917195,
    "floor": 0.0,
    "scale": 1.018110,
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def d8d9_objective(metrics: dict[str, float]) -> float:
    return float(
        35.0 * max(metrics["zero"] - 0.75, 0.0)
        + 20.0 * max(metrics["total"] - 1.52, 0.0)
        + 22.0 * max(metrics["max_decile"] - 1.43, 0.0)
        + 8.0 * max(metrics["bottom30"] - 1.18, 0.0)
        + 18.0 * max(metrics["d3"] - 1.43, 0.0)
        + 18.0 * max(0.70 - metrics["d8"], 0.0)
        + 18.0 * max(0.70 - metrics["d9"], 0.0)
        + 8.0 * max(0.80 - metrics["d10"], 0.0)
        + 4.0 * max(metrics["d10"] - 0.98, 0.0)
        + 0.12 * metrics["wape"]
    )


def read_optional_signal(path: Path | None, split: str, columns: list[str]) -> pd.DataFrame | None:
    if path is None:
        return None
    file_name = f"{'validation' if split == 'valid' else split}_{path.name}"
    file_path = path.parent / file_name
    if file_path.exists():
        return read_signal_csv(file_path, columns)
    return None


def load_cumulative(args: argparse.Namespace, frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if args.cumulative_run_dir is None:
        for threshold in (1, 2, 3, 4, 5):
            frame[f"cum_ge{threshold}_probability"] = 0.0
        return frame
    path = args.cumulative_run_dir / f"{'validation' if split == 'valid' else split}_cumulative_predictions.csv"
    columns = [f"cum_ge{threshold}_probability" for threshold in (1, 2, 3, 4, 5)]
    cumulative = read_signal_csv(path, columns)
    return frame.merge(cumulative, on=KEY_COLUMNS, how="left", validate="one_to_one")


def load_low_veto(args: argparse.Namespace, frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if args.low_veto_run_dir is None:
        frame["low_positive_veto_probability"] = 0.0
        return frame
    path = args.low_veto_run_dir / f"{'validation' if split == 'valid' else split}_low_positive_veto_predictions.csv"
    low = read_signal_csv(path, ["low_positive_veto_probability"])
    return frame.merge(low, on=KEY_COLUMNS, how="left", validate="one_to_one")


def build_lift_signals(args: argparse.Namespace, split: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame, signals, _ = build_signal_frame(args, split)
    frame = load_cumulative(args, frame, split)
    frame = load_low_veto(args, frame, split)

    for threshold in (1, 2, 3, 4, 5):
        key = f"cum_ge{threshold}"
        signals[key] = frame[f"cum_ge{threshold}_probability"].fillna(0.0).to_numpy(dtype=np.float64)
    signals["low_positive_veto"] = frame["low_positive_veto_probability"].fillna(0.0).to_numpy(dtype=np.float64)
    add_ranks(signals)
    return frame, signals


def add_ranks(signals: dict[str, np.ndarray]) -> None:
    for key in (
        "base_raw",
        "avg_raw",
        "band_expected",
        "band_tail",
        "tail",
        "veto",
        "low_positive_veto",
        "cum_ge1",
        "cum_ge2",
        "cum_ge3",
        "cum_ge4",
        "cum_ge5",
    ):
        signals[f"{key}_rank"] = rank(signals[key])


def base_892_prediction(signals: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    combined = (
        BASE_892["base"] * signals["base_raw_rank"]
        + BASE_892["avg"] * signals["avg_raw_rank"]
        + BASE_892["band"] * signals["band_expected_rank"]
        + BASE_892["tail"] * signals["tail_rank"]
        - BASE_892["veto"] * signals["veto_rank"]
    )
    cutoff = float(np.quantile(combined, BASE_892["threshold"]))
    gate = (combined >= cutoff).astype(np.float64)
    prediction = signals["regressor"] * gate * BASE_892["scale"]
    return prediction, gate


def generate_candidates(rng: np.random.Generator, trials: int) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = [
        {
            "ge3": 1.0,
            "ge4": 0.0,
            "ge5": 0.0,
            "band": 0.0,
            "tail": 0.0,
            "low": 0.8,
            "zero_weight": 0.1,
            "boost_quantile": 0.82,
            "boost": 0.12,
            "damp_quantile": 0.82,
            "damp": 0.10,
            "scale": 1.0,
        },
        {
            "ge3": 0.7,
            "ge4": 0.2,
            "ge5": 0.0,
            "band": 0.1,
            "tail": 0.0,
            "low": 0.9,
            "zero_weight": 0.1,
            "boost_quantile": 0.78,
            "boost": 0.16,
            "damp_quantile": 0.80,
            "damp": 0.14,
            "scale": 1.0,
        },
    ]
    for _ in range(trials):
        boost_weights = rng.dirichlet(np.array([3.0, 1.5, 0.7, 1.2, 0.9], dtype=np.float64))
        damp_weights = rng.dirichlet(np.array([3.2, 1.0], dtype=np.float64))
        candidates.append(
            {
                "ge3": float(boost_weights[0]),
                "ge4": float(boost_weights[1]),
                "ge5": float(boost_weights[2]),
                "band": float(boost_weights[3]),
                "tail": float(boost_weights[4]),
                "low": float(damp_weights[0]),
                "zero_weight": float(damp_weights[1]),
                "boost_quantile": float(rng.uniform(0.66, 0.93)),
                "boost": float(rng.uniform(0.02, 0.36)),
                "damp_quantile": float(rng.uniform(0.62, 0.94)),
                "damp": float(rng.uniform(0.00, 0.42)),
                "scale": float(rng.uniform(0.88, 1.03)),
            }
        )
    return candidates


def predict_lift(signals: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    base_prediction, gate = base_892_prediction(signals)
    boost_score = (
        params["ge3"] * signals["cum_ge3_rank"]
        + params["ge4"] * signals["cum_ge4_rank"]
        + params["ge5"] * signals["cum_ge5_rank"]
        + params["band"] * signals["band_tail_rank"]
        + params["tail"] * signals["tail_rank"]
    )
    damp_score = params["low"] * signals["low_positive_veto_rank"] + params["zero_weight"] * signals["veto_rank"]
    boost_cutoff = float(np.quantile(boost_score[gate > 0], params["boost_quantile"])) if np.any(gate > 0) else math.inf
    damp_cutoff = float(np.quantile(damp_score[gate > 0], params["damp_quantile"])) if np.any(gate > 0) else math.inf
    boost_gate = ((gate > 0) & (boost_score >= boost_cutoff)).astype(np.float64)
    damp_gate = ((gate > 0) & (damp_score >= damp_cutoff)).astype(np.float64)
    multiplier = (1.0 + params["boost"] * boost_gate) * (1.0 - params["damp"] * damp_gate) * params["scale"]
    return base_prediction * np.clip(multiplier, 0.25, 1.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a D8/D9 lift layer anchored on SIP candidate 892.")
    parser.add_argument("--family", choices=["SIP"], default="SIP")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-calibration-run-dir", type=Path, required=True)
    parser.add_argument("--avg-calibration-run-dir", type=Path, required=True)
    parser.add_argument("--amount-band-run-dir", type=Path, required=True)
    parser.add_argument("--cumulative-run-dir", type=Path, required=True)
    parser.add_argument("--veto-run-dir", type=Path, required=True)
    parser.add_argument("--low-veto-run-dir", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--tail-run-dir", type=Path, required=True)
    parser.add_argument("--external-signal-run-dir", type=Path, default=None)
    parser.add_argument("--tail-alpha", type=float, default=None)
    parser.add_argument("--tail-band-alpha", type=float, default=None)
    parser.add_argument("--tail-external-alpha", type=float, default=None)
    parser.add_argument("--tail-low-alpha", type=float, default=None)
    parser.add_argument("--tail-scale", type=float, default=None)
    parser.add_argument("--trials", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--screen-sample-rows", type=int, default=250000)
    parser.add_argument("--screen-top-k", type=int, default=1200)
    parser.add_argument("--top-k-test", type=int, default=300)
    return parser.parse_args()


def subset_signals(signals: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in signals.items()}


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = args.output_dir / args.family.lower() / "lift_layer_optimization" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    valid_frame, valid_signals = build_lift_signals(args, "valid")
    test_frame, test_signals = build_lift_signals(args, "test")
    valid_y = valid_frame["sales_target"].to_numpy(dtype=np.float64)
    test_y = test_frame["sales_target"].to_numpy(dtype=np.float64)
    valid_zero = valid_y <= 0.0
    test_zero = test_y <= 0.0
    valid_decile = positive_deciles(valid_y)
    test_decile = positive_deciles(test_y)

    candidates = generate_candidates(rng, args.trials)
    sample_n = min(args.screen_sample_rows, len(valid_y))
    screen_idx = np.sort(rng.choice(len(valid_y), size=sample_n, replace=False))
    screen_signals = subset_signals(valid_signals, screen_idx)
    screen_y = valid_y[screen_idx]
    screen_zero = valid_zero[screen_idx]
    screen_decile = valid_decile[screen_idx]
    rows = []
    log(f"Screening {len(candidates)} lift-layer candidates on {sample_n} validation rows")
    for idx, params in enumerate(candidates):
        prediction = predict_lift(screen_signals, params)
        metric = fast_metrics(screen_y, prediction, screen_zero, screen_decile)
        metric["lift_objective"] = d8d9_objective(metric)
        metric["candidate_id"] = idx
        metric.update(params)
        rows.append(metric)
    screen_summary = pd.DataFrame.from_records(rows).sort_values("lift_objective")
    screen_summary.to_csv(run_dir / "validation_lift_screen_candidates.csv", index=False)

    validation_rows = []
    selected_for_full = screen_summary.head(args.screen_top_k)["candidate_id"].astype(int).tolist()
    log(f"Evaluating top {len(selected_for_full)} screened candidates on full validation")
    for candidate_id in selected_for_full:
        params = candidates[candidate_id]
        prediction = predict_lift(valid_signals, params)
        metric = fast_metrics(valid_y, prediction, valid_zero, valid_decile)
        metric["lift_objective"] = d8d9_objective(metric)
        metric["candidate_id"] = candidate_id
        metric.update(params)
        validation_rows.append(metric)
    validation_summary = pd.DataFrame.from_records(validation_rows).sort_values("lift_objective")
    validation_summary.to_csv(run_dir / "validation_lift_candidates.csv", index=False)

    test_rows = []
    selected_ids = validation_summary.head(args.top_k_test)["candidate_id"].astype(int).tolist()
    for candidate_id in selected_ids:
        params = candidates[candidate_id]
        prediction = predict_lift(test_signals, params)
        metric = fast_metrics(test_y, prediction, test_zero, test_decile)
        metric["lift_objective"] = d8d9_objective(metric)
        metric["candidate_id"] = candidate_id
        metric.update(params)
        test_rows.append(metric)
    test_summary = pd.DataFrame.from_records(test_rows).sort_values("lift_objective")
    test_summary.to_csv(run_dir / "test_lift_candidates.csv", index=False)

    base_valid_pred, _ = base_892_prediction(valid_signals)
    base_test_pred, _ = base_892_prediction(test_signals)
    base_rows = [
        {"split": "validation", **fast_metrics(valid_y, base_valid_pred, valid_zero, valid_decile)},
        {"split": "test", **fast_metrics(test_y, base_test_pred, test_zero, test_decile)},
    ]
    pd.DataFrame.from_records(base_rows).to_csv(run_dir / "base_892_metrics.csv", index=False)

    selected = validation_summary.iloc[0].to_dict()
    selected_test = test_summary.loc[test_summary["candidate_id"] == int(selected["candidate_id"])].iloc[0].to_dict()
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "base_892": BASE_892,
        "base_calibration_run_dir": str(args.base_calibration_run_dir),
        "avg_calibration_run_dir": str(args.avg_calibration_run_dir),
        "amount_band_run_dir": str(args.amount_band_run_dir),
        "cumulative_run_dir": str(args.cumulative_run_dir),
        "veto_run_dir": str(args.veto_run_dir),
        "low_veto_run_dir": str(args.low_veto_run_dir),
        "base_run_dir": str(args.base_run_dir),
        "tail_run_dir": str(args.tail_run_dir),
        "trials": args.trials,
        "screen_sample_rows": args.screen_sample_rows,
        "screen_top_k": args.screen_top_k,
        "selected_validation": selected,
        "selected_test": selected_test,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Wrote lift-layer optimization outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
