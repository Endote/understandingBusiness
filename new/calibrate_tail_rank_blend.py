#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from train_stage_model import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    EncodingArtifact,
    load_dataset,
    metric_row,
    predict_frame,
)


def artifact_from_payload(payload: dict[str, object]) -> EncodingArtifact:
    return EncodingArtifact(
        numeric_features=list(payload["numeric_features"]),
        categorical_features=list(payload["categorical_features"]),
        numeric_medians={str(key): float(value) for key, value in dict(payload["numeric_medians"]).items()},
        category_maps={
            str(col): {str(level): int(code) for level, code in dict(mapping).items()}
            for col, mapping in dict(payload["category_maps"]).items()
        },
        category_sizes={str(key): int(value) for key, value in dict(payload["category_sizes"]).items()},
        feature_names=list(payload["feature_names"]),
    )


def load_regressor(run_dir: Path) -> tuple[xgb.Booster, EncodingArtifact, str]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    artifact = artifact_from_payload(json.loads((run_dir / "encoding_artifact.json").read_text(encoding="utf-8")))
    booster = xgb.Booster()
    booster.load_model(run_dir / "model.json")
    return booster, artifact, str(summary["objective"])


def rank_signal(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / max(len(values), 1)
    return ranks


def slice_masks(y_true: np.ndarray) -> dict[str, np.ndarray]:
    ranks = pd.Series(y_true).rank(method="first", pct=True).to_numpy(dtype=np.float64)
    return {
        "bottom30": ranks <= 0.30,
        "top30": ranks > 0.70,
        "top10": ranks > 0.90,
    }


def unit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(y_true.sum())
    return float(y_pred.sum() / denom) if denom else math.nan


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else math.nan


def target_metrics(y_true: np.ndarray, y_pred: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "total_ratio": unit_ratio(y_true, y_pred),
        "bottom30": unit_ratio(y_true[masks["bottom30"]], y_pred[masks["bottom30"]]),
        "top30": unit_ratio(y_true[masks["top30"]], y_pred[masks["top30"]]),
        "top10": unit_ratio(y_true[masks["top10"]], y_pred[masks["top10"]]),
    }


def adjusted_predictions(base_prediction: np.ndarray, signal_rank: np.ndarray, alpha: float, scale: float) -> np.ndarray:
    return np.clip(base_prediction * scale * np.exp(alpha * (signal_rank - 0.5)), 0.0, None)


def float_grid(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    count = int(math.floor((max_value - min_value) / step + 0.5)) + 1
    return [round(min_value + i * step, 10) for i in range(max(count, 0))]


def grid_search(
    y_true: np.ndarray,
    base_prediction: np.ndarray,
    signal_rank: np.ndarray,
    alphas: list[float],
    scales: list[float],
    bottom30_max: float,
    top10_min: float,
    target_bottom30: float,
    target_top10: float,
) -> pd.DataFrame:
    masks = slice_masks(y_true)
    rows = []
    for alpha in alphas:
        multiplier = np.exp(alpha * (signal_rank - 0.5))
        for scale in scales:
            prediction = np.clip(base_prediction * scale * multiplier, 0.0, None)
            row = target_metrics(y_true, prediction, masks)
            row.update({"alpha": alpha, "scale": scale})
            row["constraint_ok"] = bool(row["bottom30"] <= bottom30_max and row["top10"] >= top10_min)
            row["target_score"] = (
                max(0.0, row["bottom30"] - bottom30_max) * 12.0
                + max(0.0, top10_min - row["top10"]) * 12.0
                + abs(row["total_ratio"] - 1.0) * 0.9
                + row["wape"] * 0.45
                + abs(row["bottom30"] - target_bottom30) * 0.35
                + abs(row["top10"] - target_top10) * 0.35
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(["target_score", "wape", "total_ratio"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a base regressor with a tail-signal rank multiplier.")
    parser.add_argument("--family", choices=["Weeklies", "SIP"], required=True)
    parser.add_argument("--stage", choices=["regressor_positive"], default="regressor_positive")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--signal-run-dir", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=3.0)
    parser.add_argument("--alpha-max", type=float, default=5.5)
    parser.add_argument("--alpha-step", type=float, default=0.125)
    parser.add_argument("--scale-min", type=float, default=0.55)
    parser.add_argument("--scale-max", type=float, default=0.90)
    parser.add_argument("--scale-step", type=float, default=0.0125)
    parser.add_argument("--validation-bottom30-max", type=float, default=1.25)
    parser.add_argument("--validation-top10-min", type=float, default=0.60)
    parser.add_argument("--target-bottom30", type=float, default=1.20)
    parser.add_argument("--target-top10", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, manifest, _ = load_dataset(args.dataset_dir, args.family, args.stage)
    valid_df = frame.loc[frame["split"] == "valid"].copy()
    test_df = frame.loc[frame["split"] == "test"].copy()

    base_booster, base_artifact, base_objective = load_regressor(args.base_run_dir)
    signal_booster, signal_artifact, signal_objective = load_regressor(args.signal_run_dir)

    valid_base = predict_frame(base_booster, valid_df, base_artifact, base_objective)
    test_base = predict_frame(base_booster, test_df, base_artifact, base_objective)
    valid_signal_pred = predict_frame(signal_booster, valid_df, signal_artifact, signal_objective)
    test_signal_pred = predict_frame(signal_booster, test_df, signal_artifact, signal_objective)
    valid_signal_rank = rank_signal(valid_signal_pred)
    test_signal_rank = rank_signal(test_signal_pred)

    alphas = float_grid(args.alpha_min, args.alpha_max, args.alpha_step)
    scales = float_grid(args.scale_min, args.scale_max, args.scale_step)
    grid = grid_search(
        valid_df["sales_target"].to_numpy(dtype=np.float64),
        valid_base.astype(np.float64),
        valid_signal_rank,
        alphas,
        scales,
        bottom30_max=args.validation_bottom30_max,
        top10_min=args.validation_top10_min,
        target_bottom30=args.target_bottom30,
        target_top10=args.target_top10,
    )
    constrained = grid.loc[grid["constraint_ok"]]
    selected = constrained.iloc[0] if len(constrained) else grid.iloc[0]
    alpha = float(selected["alpha"])
    scale = float(selected["scale"])

    test_prediction = adjusted_predictions(test_base.astype(np.float64), test_signal_rank, alpha=alpha, scale=scale)
    valid_prediction = adjusted_predictions(valid_base.astype(np.float64), valid_signal_rank, alpha=alpha, scale=scale)
    valid_metrics = target_metrics(
        valid_df["sales_target"].to_numpy(dtype=np.float64),
        valid_prediction,
        slice_masks(valid_df["sales_target"].to_numpy(dtype=np.float64)),
    )
    test_metrics = target_metrics(
        test_df["sales_target"].to_numpy(dtype=np.float64),
        test_prediction,
        slice_masks(test_df["sales_target"].to_numpy(dtype=np.float64)),
    )

    run_dir = (
        args.output_dir
        / args.family.lower()
        / args.stage
        / "tail_rank_blend"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(run_dir / "validation_grid.csv", index=False)
    pd.DataFrame.from_records([{**test_metrics, "alpha": alpha, "scale": scale}]).to_csv(
        run_dir / "test_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "store_id": test_df["store_id"].to_numpy(),
            "product_id": test_df["product_id"].to_numpy(),
            "onsaledate": test_df["onsaledate"].to_numpy(),
            "sales_target": test_df["sales_target"].to_numpy(),
            "base_prediction": test_base,
            "signal_prediction": test_signal_pred,
            "signal_rank": test_signal_rank,
            "prediction": test_prediction,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family": args.family,
        "stage": args.stage,
        "base_run_dir": str(args.base_run_dir),
        "base_objective": base_objective,
        "signal_run_dir": str(args.signal_run_dir),
        "signal_objective": signal_objective,
        "selection": {
            "selected_from_constrained_grid": bool(len(constrained)),
            "validation_bottom30_max": args.validation_bottom30_max,
            "validation_top10_min": args.validation_top10_min,
            "target_bottom30": args.target_bottom30,
            "target_top10": args.target_top10,
            "alpha": alpha,
            "scale": scale,
            "validation_row": selected.to_dict(),
        },
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "validation_base": metric_row(
            valid_df["sales_target"].to_numpy(dtype=np.float32),
            valid_base.astype(np.float32),
            "validation_base",
        ),
        "test_base": metric_row(
            test_df["sales_target"].to_numpy(dtype=np.float32),
            test_base.astype(np.float32),
            "test_base",
        ),
        "manifest": manifest,
        "encoding_artifacts": {
            "base": asdict(base_artifact),
            "signal": asdict(signal_artifact),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {run_dir}", flush=True)
    print(pd.DataFrame.from_records([{**test_metrics, "alpha": alpha, "scale": scale}]).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
