#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(values: np.ndarray, label: str) -> dict[str, object]:
    total = float(values.sum())
    return {
        "variant": label,
        "rows": int(len(values)),
        "nonzero_rate": float((values > 0).mean()),
        "unit_sum": total,
        "mean_all": float(values.mean()),
        "mean_nonzero": float(values[values > 0].mean()) if (values > 0).any() else 0.0,
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "top10_share": float(np.sort(values)[::-1][: max(1, int(round(len(values) * 0.10)))].sum() / total) if total > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize two-tier July incidence gate from existing scored full-label columns.")
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--target-sum", type=float, default=433186.3856)
    parser.add_argument("--target-nonzero", type=float, default=0.3622)
    parser.add_argument("--target-top10-share", type=float, default=0.675)
    parser.add_argument("--scales", default="1.00,1.05,1.10,1.15,1.20,1.25")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.scoring_dir / "july_holdout_weeklies_predictions.parquet"
    frame = pd.read_parquet(path)
    base = frame["positive_regressor_prediction"].to_numpy(dtype=np.float64) * frame["amount_multiplier"].to_numpy(dtype=np.float64)
    prob = frame["incidence_raw_probability"].to_numpy(dtype=np.float64)
    rows = []
    predictions: dict[str, np.ndarray] = {}
    scales = [float(part.strip()) for part in args.scales.split(",") if part.strip()]
    for low in np.arange(0.35, 0.56, 0.025):
        for high in np.arange(0.575, 0.751, 0.025):
            if low >= high:
                continue
            for floor in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
                for global_scale in scales:
                    layer = np.where(prob >= high, 1.0, np.where(prob >= low, floor, 0.0))
                    pred = base * layer * global_scale
                    label = f"twotier_raw_low{low:.3f}_high{high:.3f}_floor{floor:.2f}_scale{global_scale:.2f}"
                    row = summarize(pred, label)
                    row["low_threshold"] = float(low)
                    row["high_threshold"] = float(high)
                    row["floor"] = float(floor)
                    row["global_scale"] = float(global_scale)
                    row["score"] = (
                        3.0 * abs(row["nonzero_rate"] - args.target_nonzero)
                        + 1.0 * abs(row["unit_sum"] / args.target_sum - 1.0)
                        + 2.0 * abs(row["top10_share"] - args.target_top10_share)
                        + 0.4 * max(row["p99"] / 19.0 - 1.0, 0.0)
                    )
                    rows.append(row)
                    predictions[label] = pred
    grid = pd.DataFrame.from_records(rows).sort_values("score")
    output_dir = args.scoring_dir / "twotier_gate_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "twotier_gate_grid.csv", index=False)
    best = grid.iloc[0].to_dict()
    best_prediction = predictions[str(best["variant"])]
    frame["july_holdout_prediction_twotier_best"] = best_prediction.astype(np.float32)
    frame.to_parquet(output_dir / "july_holdout_weeklies_predictions_twotier_best.parquet", index=False)
    frame.to_csv(output_dir / "july_holdout_weeklies_predictions_twotier_best.csv", index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_scoring_dir": str(args.scoring_dir),
        "best_variant": best,
        "outputs": {
            "grid": str(output_dir / "twotier_gate_grid.csv"),
            "predictions_parquet": str(output_dir / "july_holdout_weeklies_predictions_twotier_best.parquet"),
            "predictions_csv": str(output_dir / "july_holdout_weeklies_predictions_twotier_best.csv"),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(grid.head(20).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
