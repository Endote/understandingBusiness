#!/usr/bin/env python3
"""Plot actual-vs-predicted scatter diagnostics for saved model runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "new" / "output" / "model_runs"
PLOT_ROOT = PROJECT_ROOT / "new" / "output" / "plots" / "prediction_scatter"


@dataclass(frozen=True)
class RunInfo:
    family: str
    objective: str
    weighting: str
    run_dir: Path
    run_id: str
    summary: dict


def load_run_info(run_dir: Path) -> RunInfo | None:
    summary_path = run_dir / "run_summary.json"
    predictions_path = run_dir / "test_predictions.csv"
    if not summary_path.exists() or not predictions_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    weighting = summary.get("weighting") or "legacy"
    return RunInfo(
        family=str(summary["family"]),
        objective=str(summary["objective"]),
        weighting=str(weighting),
        run_dir=run_dir,
        run_id=run_dir.name,
        summary=summary,
    )


def discover_latest_uniform_runs() -> list[RunInfo]:
    candidates: dict[tuple[str, str], RunInfo] = {}
    for summary_path in RUN_ROOT.glob("*/*/*/*/run_summary.json"):
        info = load_run_info(summary_path.parent)
        if info is None or info.weighting != "uniform":
            continue
        key = (info.family, info.objective)
        if key not in candidates or info.run_id > candidates[key].run_id:
            candidates[key] = info

    ordered_keys = [
        ("Weeklies", "log1p_squarederror"),
        ("Weeklies", "curve_log1p"),
        ("Weeklies", "asym_curve_log1p"),
        ("Weeklies", "tweedie"),
        ("Weeklies", "quantile_50"),
        ("Weeklies", "quantile_80"),
        ("Weeklies", "quantile_90"),
        ("SIP", "log1p_squarederror"),
        ("SIP", "curve_log1p"),
        ("SIP", "asym_curve_log1p"),
        ("SIP", "tweedie"),
        ("SIP", "quantile_50"),
        ("SIP", "quantile_80"),
        ("SIP", "quantile_90"),
    ]
    return [candidates[key] for key in ordered_keys if key in candidates]


def read_predictions(run: RunInfo, sample_size: int, random_state: int) -> pd.DataFrame:
    frame = pd.read_csv(run.run_dir / "test_predictions.csv", usecols=["sales_target", "prediction"])
    frame = frame.rename(columns={"sales_target": "actual"})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[(frame["actual"] >= 0) & (frame["prediction"] >= 0)]
    if sample_size > 0 and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=random_state)
    return frame


def metric_text(run: RunInfo) -> str:
    final = run.summary["final_metrics"]
    test = final["test"]
    tail = final["test_tail_priority"]
    return (
        f"WAPE {test['wape']:.3f} | total {test['unit_ratio_pred_over_actual']:.3f} | "
        f"top10 {tail['actual_top_10_pct']['unit_ratio_pred_over_actual']:.3f}"
    )


def plot_one_axis(
    axis: plt.Axes,
    run: RunInfo,
    frame: pd.DataFrame,
    scale: str,
    axis_quantile: float,
) -> None:
    x = frame["actual"].to_numpy(dtype=float)
    y = frame["prediction"].to_numpy(dtype=float)
    if scale == "log":
        axis.set_xscale("log")
        axis.set_yscale("log")
        lower = max(0.8, min(float(x.min()), float(y.min())))
        upper = max(float(x.max()), float(y.max()))
    else:
        lower = 0.0
        qx = float(np.quantile(x, axis_quantile))
        qy = float(np.quantile(y, axis_quantile))
        upper = max(qx, qy, 1.0)

    axis.scatter(x, y, s=4, alpha=0.10, linewidths=0, color="#1f77b4", rasterized=True)
    axis.plot([lower, upper], [lower, upper], color="red", linewidth=1.4)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_title(f"{run.family} {run.objective}\n{metric_text(run)}", fontsize=10)
    axis.set_xlabel("Actual SoldQty")
    axis.set_ylabel("Predicted SoldQty")
    axis.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)

    if scale == "linear":
        max_actual = float(np.max(x))
        max_prediction = float(np.max(y))
        axis.text(
            0.02,
            0.98,
            f"axis clipped at p{axis_quantile * 100:.1f}\nmax actual {max_actual:.1f}, max pred {max_prediction:.1f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.75, "pad": 3},
        )


def plot_grid(runs: list[RunInfo], output_path: Path, scale: str, sample_size: int, axis_quantile: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 2
    rows = max(1, math.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5.5 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).ravel()
    for axis, run in zip(axes_flat, runs):
        frame = read_predictions(run, sample_size=sample_size, random_state=42)
        plot_one_axis(axis, run, frame, scale=scale, axis_quantile=axis_quantile)
    for axis in axes_flat[len(runs) :]:
        axis.axis("off")
    subtitle = "log scale, full range" if scale == "log" else f"linear scale, p{axis_quantile * 100:.1f} zoom"
    fig.suptitle(f"Actual vs Predicted SoldQty - latest uniform model runs ({subtitle})", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_individuals(runs: list[RunInfo], output_dir: Path, scale: str, sample_size: int, axis_quantile: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for run in runs:
        frame = read_predictions(run, sample_size=sample_size, random_state=42)
        fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        plot_one_axis(axis, run, frame, scale=scale, axis_quantile=axis_quantile)
        fig.suptitle(f"Actual vs Predicted SoldQty - {run.family} {run.objective} {run.run_id}", fontsize=13)
        path = output_dir / f"{run.family.lower()}_{run.objective}_{run.run_id}_{scale}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200_000, help="Maximum rows plotted per run; 0 means all rows.")
    parser.add_argument("--axis-quantile", type=float, default=0.999, help="Linear plot axis upper quantile.")
    parser.add_argument("--skip-individual", action="store_true", help="Only write combined grids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_latest_uniform_runs()
    if not runs:
        raise SystemExit("No latest uniform runs found")

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    written = [
        PLOT_ROOT / "latest_uniform_actual_vs_predicted_linear.png",
        PLOT_ROOT / "latest_uniform_actual_vs_predicted_log.png",
    ]
    plot_grid(runs, written[0], scale="linear", sample_size=args.sample_size, axis_quantile=args.axis_quantile)
    plot_grid(runs, written[1], scale="log", sample_size=args.sample_size, axis_quantile=args.axis_quantile)
    if not args.skip_individual:
        written.extend(plot_individuals(runs, PLOT_ROOT / "individual", "linear", args.sample_size, args.axis_quantile))
        written.extend(plot_individuals(runs, PLOT_ROOT / "individual", "log", args.sample_size, args.axis_quantile))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
