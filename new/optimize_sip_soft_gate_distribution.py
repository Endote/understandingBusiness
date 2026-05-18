#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_sip_ceiled_group_alignment import DEFAULT_DB_URI, actual_query, actual_units_query, decile_table, run_psql_csv, series_summary


TARGET_GROUPS = {
    "segment": ["segment"],
    "subsegment": ["subsegment"],
    "classoftrade": ["classoftrade"],
    "store_chain": ["store_chain"],
}
CALIBRATION_GROUPS = {
    "none": [],
    "segment": ["segment"],
    "subsegment": ["subsegment"],
    "segment_classoftrade": ["segment", "classoftrade"],
    "subsegment_classoftrade": ["subsegment", "classoftrade"],
}
BUDGET_STRATEGIES = {
    "none": [],
    "segment": ["segment"],
    "segment_subsegment": ["segment", "subsegment"],
    "segment_subsegment_class": ["segment", "subsegment", "classoftrade"],
}
SIGNAL_COLUMNS = [
    "incidence_base_raw_probability",
    "incidence_avg_raw_probability",
    "band_positive_probability",
    "band_expected_units",
    "cum_ge1_probability",
    "cum_ge2_probability",
    "cum_ge3_probability",
    "cum_ge4_probability",
    "tail_probability",
    "band_tail_probability",
    "positive_regressor_prediction",
    "positive_q40_prediction",
    "positive_tweedie_signal",
    "zero_veto_probability",
    "low_positive_veto_probability",
    "sip_892_rank",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def key_series(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(["__all__"] * len(frame), index=frame.index)
    return frame[cols].astype("string").fillna("__NA__").agg(" | ".join, axis=1)


def actual_decile_targets(actual_units: np.ndarray) -> dict[str, float]:
    table = decile_table(actual_units, "actual", "actual")
    return {
        "nonzero_rate": float((actual_units > 0).mean()),
        "mean_all": float(np.mean(actual_units)),
        "mean_nonzero": float(actual_units[actual_units > 0].mean()),
        **{f"d{int(row.decile_num)}_mean": float(row.mean_units) for row in table.itertuples(index=False)},
        **{f"d{int(row.decile_num)}_median": float(row.median_units) for row in table.itertuples(index=False)},
        **{f"d{int(row.decile_num)}_share": float(row.unit_share) for row in table.itertuples(index=False)},
    }


def actual_group_share(db_uri: str, cols: list[str]) -> pd.DataFrame:
    actual = run_psql_csv(db_uri, actual_query(cols))
    actual["actual_unit_share"] = actual["actual_units"] / actual["actual_units"].sum()
    actual["group_key"] = key_series(actual, cols)
    return actual


def prediction_decile_metrics(values: np.ndarray) -> dict[str, float]:
    table = decile_table(values, "prediction", "prediction")
    result = {
        "nonzero_rate": float((values > 0).mean()),
        "unit_sum": float(values.sum()),
        "mean_all": float(values.mean()),
        "mean_nonzero": float(values[values > 0].mean()) if np.any(values > 0) else 0.0,
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }
    for row in table.itertuples(index=False):
        decile = int(row.decile_num)
        result[f"d{decile}_mean"] = float(row.mean_units)
        result[f"d{decile}_median"] = float(row.median_units)
        result[f"d{decile}_share"] = float(row.unit_share)
        result[f"d{decile}_zero_frac"] = float(row.zero_row_frac)
    return result


def build_group_info(frame: pd.DataFrame, actual_frame: pd.DataFrame, cols: list[str]) -> dict[str, object]:
    keys = key_series(frame, cols)
    codes, categories = pd.factorize(keys, sort=False)
    target_share = target_share_series(actual_frame)
    category_index = pd.Index(categories.astype(str))
    target_for_categories = target_share.reindex(category_index).fillna(0.0).to_numpy(dtype=np.float64)
    target_available = target_for_categories.copy()
    available_sum = float(target_available.sum())
    if available_sum > 0:
        target_available /= available_sum
    missing_target_mass = float(target_share.loc[~target_share.index.isin(category_index)].sum())
    return {
        "codes": codes.astype(np.int64),
        "categories": category_index,
        "target_share": target_for_categories,
        "target_share_available": target_available,
        "missing_target_mass": missing_target_mass,
        "n_categories": int(len(category_index)),
    }


def group_tvd_fast(prediction: np.ndarray, group_infos: dict[str, dict[str, object]]) -> dict[str, float]:
    total = float(prediction.sum())
    result: dict[str, float] = {}
    for name, info in group_infos.items():
        codes = info["codes"]  # type: ignore[assignment]
        n_categories = int(info["n_categories"])
        target = info["target_share"]  # type: ignore[assignment]
        if total <= 0:
            predicted_share = np.zeros(n_categories, dtype=np.float64)
        else:
            predicted_share = np.bincount(codes, weights=prediction, minlength=n_categories).astype(np.float64) / total
        diff = np.abs(predicted_share - target)
        result[f"{name}_tvd"] = float(0.5 * (diff.sum() + float(info["missing_target_mass"])))
        result[f"{name}_max_shift_pp"] = float(100.0 * diff.max()) if len(diff) else 0.0
    return result


def group_unit_shares(frame: pd.DataFrame, prediction: np.ndarray, cols: list[str]) -> pd.Series:
    total = float(prediction.sum())
    if total <= 0:
        return pd.Series(dtype=np.float64)
    keys = key_series(frame, cols)
    return pd.Series(prediction).groupby(keys, sort=False).sum() / total


def apply_group_calibration(
    prediction: np.ndarray,
    group_info: dict[str, object],
    strength: float,
    min_mult: float,
    max_mult: float,
    target_key: str = "target_share",
) -> np.ndarray:
    if prediction.sum() <= 0:
        return prediction
    before_total = float(prediction.sum())
    codes = group_info["codes"]  # type: ignore[assignment]
    n_categories = int(group_info["n_categories"])
    target = group_info[target_key]  # type: ignore[assignment]
    pred_share = np.bincount(codes, weights=prediction, minlength=n_categories).astype(np.float64) / before_total
    target_safe = np.where(target > 0, target, pred_share)
    ratio = target_safe / np.clip(pred_share, 1e-9, None)
    multiplier_by_group = np.clip(1.0 + strength * (ratio - 1.0), min_mult, max_mult)
    row_multiplier = multiplier_by_group[codes]
    calibrated = prediction * row_multiplier
    after_total = float(calibrated.sum())
    if after_total > 0:
        calibrated *= before_total / after_total
    return calibrated


def apply_share_cap(prediction: np.ndarray, group_info: dict[str, object], group_name: str, cap_share: float) -> np.ndarray:
    total = float(prediction.sum())
    if total <= 0 or cap_share <= 0:
        return prediction
    categories = group_info["categories"]  # type: ignore[assignment]
    matches = np.flatnonzero(categories.to_numpy(dtype=str) == group_name)
    if len(matches) == 0:
        return prediction
    codes = group_info["codes"]  # type: ignore[assignment]
    mask = codes == int(matches[0])
    group_sum = float(prediction[mask].sum())
    cap_sum = cap_share * total
    if group_sum <= cap_sum or group_sum <= 0:
        return prediction
    capped = prediction.copy()
    removed = group_sum - cap_sum
    capped[mask] *= cap_sum / group_sum
    other_sum = float(capped[~mask].sum())
    if other_sum > 0:
        capped[~mask] *= (other_sum + removed) / other_sum
    return capped


def apply_ceiled_share_cap(prediction: np.ndarray, group_info: dict[str, object], group_name: str, cap_share: float) -> np.ndarray:
    if cap_share <= 0 or cap_share >= 1:
        return prediction
    ceiled = np.ceil(prediction)
    total = float(ceiled.sum())
    if total <= 0:
        return prediction
    categories = group_info["categories"]  # type: ignore[assignment]
    matches = np.flatnonzero(categories.to_numpy(dtype=str) == group_name)
    if len(matches) == 0:
        return prediction
    codes = group_info["codes"]  # type: ignore[assignment]
    mask = (codes == int(matches[0])) & (ceiled > 0)
    group_units = float(ceiled[mask].sum())
    if group_units <= cap_share * total:
        return prediction
    required_removed = (group_units - cap_share * total) / max(1.0 - cap_share, 1e-9)
    candidate_idx = np.flatnonzero(mask)
    if len(candidate_idx) == 0:
        return prediction
    order = np.argsort(prediction[candidate_idx], kind="mergesort")
    ordered_idx = candidate_idx[order]
    cumulative_removed = np.cumsum(ceiled[ordered_idx])
    remove_count = int(np.searchsorted(cumulative_removed, required_removed, side="left") + 1)
    capped = prediction.copy()
    capped[ordered_idx[:remove_count]] = 0.0
    return capped


def apply_ceiled_share_caps(prediction: np.ndarray, group_infos: dict[str, dict[str, object]], params: dict[str, float | str]) -> np.ndarray:
    capped = prediction
    for _ in range(int(params.get("ceil_cap_iterations", 1))):
        capped = apply_ceiled_share_cap(capped, group_infos["segment"], "PUZZLES", float(params.get("ceil_puzzles_cap", 1.0)))
        capped = apply_ceiled_share_cap(capped, group_infos["subsegment"], "SUDOKU", float(params.get("ceil_sudoku_cap", 1.0)))
        capped = apply_ceiled_share_cap(capped, group_infos["subsegment"], "WORD SEEK", float(params.get("ceil_word_seek_cap", 1.0)))
    return capped


def apply_budget_allocator(prediction: np.ndarray, group_infos: dict[str, dict[str, object]], params: dict[str, float | str]) -> np.ndarray:
    strategy = str(params.get("budget_strategy", "none"))
    groups = BUDGET_STRATEGIES[strategy]
    if not groups or float(prediction.sum()) <= 0:
        return prediction
    allocated = prediction
    iterations = int(params.get("budget_iterations", 2))
    for _ in range(iterations):
        for group in groups:
            allocated = apply_group_calibration(
                allocated,
                group_infos[group],
                float(params.get("budget_strength", 0.50)),
                float(params.get("budget_min", 0.25)),
                float(params.get("budget_max", 2.50)),
                target_key="target_share_available",
            )
        if float(params.get("puzzles_cap", 1.0)) < 1.0:
            allocated = apply_share_cap(allocated, group_infos["segment"], "PUZZLES", float(params["puzzles_cap"]))
        if float(params.get("sudoku_cap", 1.0)) < 1.0:
            allocated = apply_share_cap(allocated, group_infos["subsegment"], "SUDOKU", float(params["sudoku_cap"]))
        if float(params.get("word_seek_cap", 1.0)) < 1.0:
            allocated = apply_share_cap(allocated, group_infos["subsegment"], "WORD SEEK", float(params["word_seek_cap"]))
    return allocated


def target_share_series(actual_share_frame: pd.DataFrame) -> pd.Series:
    return actual_share_frame.set_index("group_key")["actual_unit_share"]


def group_share_from_info(prediction: np.ndarray, group_info: dict[str, object], group_name: str) -> float:
    total = float(prediction.sum())
    if total <= 0:
        return 0.0
    categories = group_info["categories"]  # type: ignore[assignment]
    matches = np.flatnonzero(categories.to_numpy(dtype=str) == group_name)
    if len(matches) == 0:
        return 0.0
    codes = group_info["codes"]  # type: ignore[assignment]
    return float(prediction[codes == int(matches[0])].sum() / total)


def candidate_score(metrics: dict[str, float], targets: dict[str, float], profile: str) -> float:
    if profile == "integer_ratio_groups":
        d8_cap = 1.40 * targets["d8_mean"]
        d9_floor = 0.80 * targets["d9_mean"]
        d9_cap = 1.30 * targets["d9_mean"]
        d10_floor = 0.95 * targets["d10_mean"]
        d10_cap = 1.35 * targets["d10_mean"]
        return float(
            7.0 * abs(metrics["ceil_nonzero_rate"] - targets["nonzero_rate"])
            + 12.0 * max(metrics["ceil_nonzero_rate"] - 0.30, 0.0) ** 2
            + 14.0 * max(metrics["ceil_d8_mean"] - d8_cap, 0.0)
            + 4.0 * max(0.25 - metrics["ceil_d8_mean"], 0.0)
            + 9.0 * max(d9_floor - metrics["ceil_d9_mean"], 0.0)
            + 6.0 * max(metrics["ceil_d9_mean"] - d9_cap, 0.0)
            + 7.0 * max(d10_floor - metrics["ceil_d10_mean"], 0.0)
            + 4.0 * max(metrics["ceil_d10_mean"] - d10_cap, 0.0)
            + 15.0 * metrics["ceil_segment_tvd"]
            + 12.0 * metrics["ceil_subsegment_tvd"]
            + 8.0 * metrics["ceil_classoftrade_tvd"]
            + 5.0 * metrics["ceil_store_chain_tvd"]
            + 50.0 * max(metrics["ceil_puzzles_share"] - 0.32, 0.0)
            + 38.0 * max(metrics["ceil_sudoku_share"] - 0.13, 0.0)
            + 34.0 * max(metrics["raw_word_seek_share"] - 0.16, 0.0)
            + 5.0 * abs(metrics["ceil_mean_nonzero"] - targets["mean_nonzero"])
            + 3.0 * max(targets["mean_all"] - metrics["raw_mean_all"], 0.0)
        )
    if profile == "ceil_balanced":
        return float(
            4.0 * abs(metrics["ceil_nonzero_rate"] - targets["nonzero_rate"])
            + 120.0 * max(metrics["ceil_nonzero_rate"] - 0.35, 0.0) ** 2
            + 10.0 * abs(metrics["ceil_d8_mean"] - 0.75)
            + 12.0 * abs(metrics["ceil_d9_mean"] - 1.10)
            + 5.0 * abs(metrics["ceil_d10_mean"] - 3.25)
            + 8.0 * metrics["ceil_segment_tvd"]
            + 7.0 * metrics["ceil_subsegment_tvd"]
            + 5.0 * metrics["ceil_classoftrade_tvd"]
            + 3.0 * metrics["ceil_store_chain_tvd"]
            + 28.0 * max(metrics["ceil_puzzles_share"] - 0.36, 0.0)
            + 25.0 * max(metrics["ceil_sudoku_share"] - 0.14, 0.0)
            + 8.0 * max(metrics["ceil_mean_nonzero"] - 2.75, 0.0)
            + 4.0 * abs(metrics["raw_mean_all"] - targets["mean_all"])
        )
    if profile == "budget_groups":
        return float(
            5.0 * abs(metrics["ceil_nonzero_rate"] - targets["nonzero_rate"])
            + 180.0 * max(metrics["ceil_nonzero_rate"] - 0.35, 0.0) ** 2
            + 8.0 * abs(metrics["ceil_d8_mean"] - 0.75)
            + 9.0 * abs(metrics["ceil_d9_mean"] - 1.10)
            + 4.0 * abs(metrics["ceil_d10_mean"] - 3.25)
            + 18.0 * metrics["ceil_segment_tvd"]
            + 14.0 * metrics["ceil_subsegment_tvd"]
            + 8.0 * metrics["ceil_classoftrade_tvd"]
            + 4.0 * metrics["ceil_store_chain_tvd"]
            + 55.0 * max(metrics["ceil_puzzles_share"] - 0.34, 0.0)
            + 40.0 * max(metrics["ceil_sudoku_share"] - 0.13, 0.0)
            + 35.0 * max(metrics["raw_word_seek_share"] - 0.16, 0.0)
            + 8.0 * max(metrics["ceil_mean_nonzero"] - 2.75, 0.0)
            + 2.0 * max(targets["mean_all"] - metrics["raw_mean_all"], 0.0)
        )
    if profile == "strict_groups":
        return float(
            7.0 * abs(metrics["raw_nonzero_rate"] - targets["nonzero_rate"])
            + 9.0 * abs(metrics["raw_d8_mean"] - targets["d8_mean"])
            + 10.0 * abs(metrics["raw_d9_mean"] - targets["d9_mean"])
            + 3.0 * abs(metrics["raw_d10_mean"] - targets["d10_mean"])
            + 15.0 * metrics["raw_segment_tvd"]
            + 12.0 * metrics["raw_subsegment_tvd"]
            + 8.0 * metrics["raw_classoftrade_tvd"]
            + 4.0 * metrics["raw_store_chain_tvd"]
            + 45.0 * max(metrics["raw_puzzles_share"] - 0.32, 0.0)
            + 35.0 * max(metrics["raw_sudoku_share"] - 0.12, 0.0)
            + 30.0 * max(metrics["raw_word_seek_share"] - 0.16, 0.0)
            + 8.0 * max(metrics["raw_mean_nonzero"] - 4.5, 0.0)
            + 4.0 * max(metrics["ceil_nonzero_rate"] - 0.30, 0.0)
            + 4.0 * abs(metrics["ceil_d8_mean"] - max(0.50, targets["d8_median"]))
            + 4.0 * abs(metrics["ceil_d9_mean"] - max(1.00, targets["d9_median"]))
            + 1.0 * abs(metrics["ceil_d10_mean"] - max(3.00, targets["d10_median"]))
        )
    return float(
        9.0 * abs(metrics["raw_nonzero_rate"] - targets["nonzero_rate"])
        + 12.0 * abs(metrics["raw_d8_mean"] - targets["d8_mean"])
        + 13.0 * abs(metrics["raw_d9_mean"] - targets["d9_mean"])
        + 4.0 * abs(metrics["raw_d10_mean"] - targets["d10_mean"])
        + 7.0 * abs(metrics["raw_d10_share"] - targets["d10_share"])
        + 7.0 * metrics["raw_segment_tvd"]
        + 6.0 * metrics["raw_subsegment_tvd"]
        + 4.0 * metrics["raw_classoftrade_tvd"]
        + 2.5 * metrics["raw_store_chain_tvd"]
        + 20.0 * max(metrics["raw_puzzles_share"] - 0.35, 0.0)
        + 20.0 * max(metrics["raw_sudoku_share"] - 0.12, 0.0)
        + 8.0 * max(metrics["raw_mean_nonzero"] - 5.0, 0.0)
        + 5.0 * max(metrics["ceil_nonzero_rate"] - 0.30, 0.0)
        + 6.0 * abs(metrics["ceil_d8_mean"] - max(0.50, targets["d8_median"]))
        + 6.0 * abs(metrics["ceil_d9_mean"] - max(1.00, targets["d9_median"]))
        + 2.0 * abs(metrics["ceil_d10_mean"] - max(3.00, targets["d10_median"]))
    )


def build_score(signals: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    score = (
        params["incidence_base"] * signals["incidence_base_raw_probability_rank"]
        + params["incidence_avg"] * signals["incidence_avg_raw_probability_rank"]
        + params["band_expected"] * signals["band_expected_units_rank"]
        + params["band_positive"] * signals["band_positive_probability_rank"]
        + params["ge1"] * signals["cum_ge1_probability_rank"]
        + params["ge2"] * signals["cum_ge2_probability_rank"]
        + params["ge3"] * signals["cum_ge3_probability_rank"]
        + params["tail"] * signals["tail_probability_rank"]
        + params["band_tail"] * signals["band_tail_probability_rank"]
        + params["amount"] * signals["positive_regressor_prediction_rank"]
        - params["zero_veto"] * signals["zero_veto_probability_rank"]
        - params["low_veto"] * signals["low_positive_veto_probability_rank"]
    )
    return rank(score)


def build_prediction(signals: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    score = build_score(signals, params)
    amount = signals["positive_regressor_prediction"].astype(np.float64)
    low_cut = params["low_quantile"]
    high_cut = params["high_quantile"]
    gate_layer = np.where(score >= high_cut, 1.0, np.where(score >= low_cut, params["mid_floor"], params["low_floor"]))
    d8d9_layer = 1.0 + params["d8d9_boost"] * ((score >= low_cut) & (score < high_cut) & (signals["d8d9_score_rank"] >= params["d8d9_cut"]))
    top_layer = 1.0 - params["top_damp"] * ((score >= high_cut) & (signals["top_score_rank"] >= params["top_cut"]))
    low_layer = 1.0 - params["low_damp"] * ((score >= low_cut) & (signals["low_score_rank"] >= params["low_cut"]))
    prediction = amount * gate_layer * d8d9_layer * top_layer * low_layer * params["global_scale"]
    return np.clip(prediction, 0.0, None)


def deterministic_candidates() -> list[dict[str, float | str]]:
    seeds = [
        {"variant_family": "weeklies_like", "low_quantile": 0.72, "high_quantile": 0.90, "mid_floor": 0.20, "global_scale": 0.80},
        {"variant_family": "sip_sparse_soft", "low_quantile": 0.78, "high_quantile": 0.915, "mid_floor": 0.14, "global_scale": 0.78},
        {"variant_family": "sip_d8d9_fill", "low_quantile": 0.75, "high_quantile": 0.90, "mid_floor": 0.18, "global_scale": 0.72},
        {"variant_family": "sip_conservative", "low_quantile": 0.82, "high_quantile": 0.93, "mid_floor": 0.12, "global_scale": 0.82},
    ]
    rows: list[dict[str, float | str]] = []
    for seed in seeds:
        for calibration_group in ("none", "segment", "subsegment", "segment_classoftrade", "subsegment_classoftrade"):
            for strength in (0.0, 0.20, 0.35, 0.50):
                rows.append(
                    {
                        **seed,
                        "calibration_group": calibration_group,
                        "calibration_strength": strength,
                        "calibration_min": 0.55,
                        "calibration_max": 1.65,
                        "incidence_base": 0.35,
                        "incidence_avg": 0.08,
                        "band_expected": 0.20,
                        "band_positive": 0.05,
                        "ge1": 0.10,
                        "ge2": 0.08,
                        "ge3": 0.07,
                        "tail": 0.04,
                        "band_tail": 0.03,
                        "amount": 0.08,
                        "zero_veto": 0.03,
                        "low_veto": 0.07,
                        "low_floor": 0.0,
                        "d8d9_boost": 0.10,
                        "d8d9_cut": 0.62,
                        "top_damp": 0.20,
                        "top_cut": 0.88,
                        "low_damp": 0.15,
                        "low_cut": 0.80,
                        "budget_strategy": "none",
                        "budget_strength": 0.0,
                        "budget_iterations": 0,
                        "budget_min": 0.40,
                        "budget_max": 2.20,
                        "puzzles_cap": 1.0,
                        "sudoku_cap": 1.0,
                        "word_seek_cap": 1.0,
                        "ceil_puzzles_cap": 1.0,
                        "ceil_sudoku_cap": 1.0,
                        "ceil_word_seek_cap": 1.0,
                        "ceil_cap_iterations": 0,
                    }
                )
    budget_seed_ids = list(rows[:])
    for row in budget_seed_ids:
        for strategy in ("segment", "segment_subsegment", "segment_subsegment_class"):
            for strength in (0.45, 0.65, 0.85):
                rows.append(
                    {
                        **row,
                        "variant_family": f"{row['variant_family']}_budget",
                        "budget_strategy": strategy,
                        "budget_strength": strength,
                        "budget_iterations": 3,
                        "budget_min": 0.20,
                        "budget_max": 3.50,
                        "puzzles_cap": 0.36,
                        "sudoku_cap": 0.14,
                        "word_seek_cap": 0.18,
                        "ceil_puzzles_cap": 1.0,
                        "ceil_sudoku_cap": 1.0,
                        "ceil_word_seek_cap": 1.0,
                        "ceil_cap_iterations": 0,
                    }
                )
    return rows


def random_candidates(rng: np.random.Generator, trials: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    group_names = tuple(CALIBRATION_GROUPS)
    budget_names = tuple(BUDGET_STRATEGIES)
    family_names = ("soft_rank", "d8d9_fill", "segment_constrained", "ceil_aware", "sparse_soft")
    for _ in range(trials):
        weights = rng.dirichlet(np.array([4.0, 1.2, 2.2, 0.7, 1.3, 1.1, 0.9, 0.6, 0.5, 1.0], dtype=np.float64))
        veto_weights = rng.dirichlet(np.array([1.0, 2.0], dtype=np.float64))
        low_q = float(rng.uniform(0.68, 0.86))
        high_q = float(rng.uniform(max(low_q + 0.055, 0.84), 0.955))
        rows.append(
            {
                "variant_family": str(rng.choice(family_names)),
                "calibration_group": str(rng.choice(group_names, p=np.array([0.15, 0.25, 0.25, 0.20, 0.15]))),
                "calibration_strength": float(rng.choice([0.0, 0.15, 0.25, 0.35, 0.50, 0.65])),
                "calibration_min": float(rng.choice([0.40, 0.55, 0.65, 0.75])),
                "calibration_max": float(rng.choice([1.25, 1.45, 1.65, 1.90, 2.20])),
                "incidence_base": float(weights[0]),
                "incidence_avg": float(weights[1]),
                "band_expected": float(weights[2]),
                "band_positive": float(weights[3]),
                "ge1": float(weights[4]),
                "ge2": float(weights[5]),
                "ge3": float(weights[6]),
                "tail": float(weights[7]),
                "band_tail": float(weights[8]),
                "amount": float(weights[9]),
                "zero_veto": float(0.05 * veto_weights[0]),
                "low_veto": float(0.12 * veto_weights[1]),
                "low_quantile": low_q,
                "high_quantile": high_q,
                "mid_floor": float(rng.uniform(0.06, 0.32)),
                "low_floor": float(rng.choice([0.0, 0.0, 0.0, 0.02, 0.04])),
                "global_scale": float(rng.uniform(0.48, 1.08)),
                "d8d9_boost": float(rng.uniform(0.00, 0.28)),
                "d8d9_cut": float(rng.uniform(0.52, 0.82)),
                "top_damp": float(rng.uniform(0.00, 0.48)),
                "top_cut": float(rng.uniform(0.78, 0.96)),
                "low_damp": float(rng.uniform(0.00, 0.40)),
                "low_cut": float(rng.uniform(0.64, 0.92)),
                "budget_strategy": str(rng.choice(budget_names, p=np.array([0.45, 0.22, 0.22, 0.11]))),
                "budget_strength": float(rng.uniform(0.25, 0.95)),
                "budget_iterations": int(rng.choice([1, 2, 3, 4])),
                "budget_min": float(rng.choice([0.12, 0.20, 0.30, 0.45])),
                "budget_max": float(rng.choice([1.80, 2.40, 3.20, 4.00])),
                "puzzles_cap": float(rng.choice([1.00, 0.50, 0.42, 0.36, 0.32])),
                "sudoku_cap": float(rng.choice([1.00, 0.22, 0.18, 0.14, 0.12])),
                "word_seek_cap": float(rng.choice([1.00, 0.26, 0.22, 0.18, 0.16])),
                "ceil_puzzles_cap": float(rng.choice([1.00, 0.42, 0.36, 0.32, 0.28])),
                "ceil_sudoku_cap": float(rng.choice([1.00, 0.18, 0.15, 0.13, 0.11])),
                "ceil_word_seek_cap": float(rng.choice([1.00, 0.22, 0.18, 0.16, 0.14])),
                "ceil_cap_iterations": int(rng.choice([0, 1, 2, 3], p=np.array([0.35, 0.35, 0.20, 0.10]))),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize SIP soft-gate and group-calibrated July holdout distributions.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--screen-sample-rows", type=int, default=150000)
    parser.add_argument("--screen-top-k", type=int, default=1200)
    parser.add_argument("--objective-profile", choices=["balanced", "strict_groups", "ceil_balanced", "budget_groups", "integer_ratio_groups"], default="balanced")
    return parser.parse_args()


def evaluate_candidate(
    frame: pd.DataFrame,
    signals: dict[str, np.ndarray],
    params: dict[str, float | str],
    group_infos: dict[str, dict[str, object]],
    targets: dict[str, float],
    candidate_id: int,
    objective_profile: str,
) -> tuple[dict[str, float | str], np.ndarray]:
    prediction = build_prediction(signals, params)  # type: ignore[arg-type]
    calibration_group = str(params["calibration_group"])
    group_cols = CALIBRATION_GROUPS[calibration_group]
    if group_cols and float(params["calibration_strength"]) > 0:
        prediction = apply_group_calibration(
            prediction,
            group_infos[calibration_group],
            float(params["calibration_strength"]),
            float(params["calibration_min"]),
            float(params["calibration_max"]),
        )
    prediction = apply_budget_allocator(prediction, group_infos, params)
    prediction = apply_ceiled_share_caps(prediction, group_infos, params)
    ceil_prediction = np.ceil(prediction)
    raw = prediction_decile_metrics(prediction)
    ceil = prediction_decile_metrics(ceil_prediction)
    raw_groups = group_tvd_fast(prediction, {name: group_infos[name] for name in TARGET_GROUPS})
    ceil_groups = group_tvd_fast(ceil_prediction, {name: group_infos[name] for name in TARGET_GROUPS})
    metrics: dict[str, float | str] = {
        "candidate_id": candidate_id,
        **params,
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"ceil_{key}": value for key, value in ceil.items()},
        **{f"raw_{key}": value for key, value in raw_groups.items()},
        **{f"ceil_{key}": value for key, value in ceil_groups.items()},
        "raw_puzzles_share": group_share_from_info(prediction, group_infos["segment"], "PUZZLES"),
        "raw_sudoku_share": group_share_from_info(prediction, group_infos["subsegment"], "SUDOKU"),
        "raw_word_seek_share": group_share_from_info(prediction, group_infos["subsegment"], "WORD SEEK"),
        "ceil_puzzles_share": group_share_from_info(ceil_prediction, group_infos["segment"], "PUZZLES"),
        "ceil_sudoku_share": group_share_from_info(ceil_prediction, group_infos["subsegment"], "SUDOKU"),
    }
    metrics["distribution_score"] = candidate_score(metrics, targets, objective_profile)  # type: ignore[arg-type]
    return metrics, prediction


def write_candidate_artifact(
    source_dir: Path,
    output_root: Path,
    frame: pd.DataFrame,
    prediction: np.ndarray,
    row: dict[str, float | str],
    name: str,
) -> Path:
    candidate_dir = output_root / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["sip_prediction_soft_distribution"] = prediction.astype(np.float32)
    output["july_holdout_prediction"] = prediction.astype(np.float32)
    parquet_path = candidate_dir / "july_holdout_sip_predictions.parquet"
    csv_path = candidate_dir / "july_holdout_sip_predictions.csv"
    output.to_parquet(parquet_path, index=False)
    output.to_csv(csv_path, index=False)
    for filename in ("missing_holdout_embeddings.csv",):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, candidate_dir / filename)
    pd.DataFrame.from_records([series_summary(prediction, "july_holdout_prediction")]).to_csv(candidate_dir / "prediction_summary.csv", index=False)
    (candidate_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_scoring_dir": str(source_dir),
                "candidate_name": name,
                "candidate": row,
                "prediction_column": "july_holdout_prediction",
                "outputs": {
                    "predictions_parquet": str(parquet_path),
                    "predictions_csv": str(csv_path),
                    "prediction_summary": str(candidate_dir / "prediction_summary.csv"),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return candidate_dir


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_root = args.output_dir or (args.scoring_dir / "soft_gate_distribution_grid" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    log("Reading current SIP July scoring signals")
    frame = pd.read_parquet(args.scoring_dir / "july_holdout_sip_predictions.parquet")
    signals = {column: frame[column].fillna(0.0).to_numpy(dtype=np.float64) for column in SIGNAL_COLUMNS}
    for column in SIGNAL_COLUMNS:
        signals[f"{column}_rank"] = rank(signals[column])
    signals["d8d9_score_rank"] = rank(
        0.55 * signals["cum_ge1_probability_rank"]
        + 0.35 * signals["cum_ge2_probability_rank"]
        - 0.25 * signals["low_positive_veto_probability_rank"]
    )
    signals["top_score_rank"] = rank(
        0.60 * signals["positive_regressor_prediction_rank"]
        + 0.30 * signals["tail_probability_rank"]
        + 0.10 * signals["cum_ge3_probability_rank"]
    )
    signals["low_score_rank"] = rank(0.75 * signals["low_positive_veto_probability_rank"] + 0.25 * signals["zero_veto_probability_rank"])

    log("Reading current full-labeled SIP actual targets from Postgres")
    actual_units = run_psql_csv(args.db_uri, actual_units_query())["actual_units"].to_numpy(dtype=np.float64)
    targets = actual_decile_targets(actual_units)
    actual_shares = {name: actual_group_share(args.db_uri, cols) for name, cols in TARGET_GROUPS.items()}
    calibration_actuals = {
        "none": pd.DataFrame({"group_key": [], "actual_unit_share": []}),
        "segment": actual_shares["segment"],
        "subsegment": actual_shares["subsegment"],
        "segment_classoftrade": actual_group_share(args.db_uri, ["segment", "classoftrade"]),
        "subsegment_classoftrade": actual_group_share(args.db_uri, ["subsegment", "classoftrade"]),
    }
    group_infos = {
        **{name: build_group_info(frame, actual_shares[name], cols) for name, cols in TARGET_GROUPS.items()},
        **{
            name: build_group_info(frame, calibration_actuals[name], cols)
            for name, cols in CALIBRATION_GROUPS.items()
            if name != "none"
        },
    }
    (run_root / "actual_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    candidates = deterministic_candidates() + random_candidates(rng, args.trials)
    sample_n = min(args.screen_sample_rows, len(frame))
    sample_idx = np.sort(rng.choice(len(frame), size=sample_n, replace=False))
    sample_frame = frame.iloc[sample_idx].reset_index(drop=True)
    sample_signals = {key: value[sample_idx] for key, value in signals.items()}
    sample_group_infos = {
        **{name: build_group_info(sample_frame, actual_shares[name], cols) for name, cols in TARGET_GROUPS.items()},
        **{
            name: build_group_info(sample_frame, calibration_actuals[name], cols)
            for name, cols in CALIBRATION_GROUPS.items()
            if name != "none"
        },
    }

    log(f"Screening {len(candidates)} soft-gate distribution candidates on {sample_n} rows")
    screen_rows: list[dict[str, float | str]] = []
    for candidate_id, params in enumerate(candidates):
        metrics, _ = evaluate_candidate(sample_frame, sample_signals, params, sample_group_infos, targets, candidate_id, args.objective_profile)
        screen_rows.append(metrics)
        if (candidate_id + 1) % 2500 == 0:
            log(f"Screened {candidate_id + 1} candidates")
    screen_grid = pd.DataFrame.from_records(screen_rows).sort_values("distribution_score")
    screen_grid.to_csv(run_root / "soft_gate_distribution_screen_grid.csv", index=False)

    selected_for_full = screen_grid.head(args.screen_top_k)["candidate_id"].astype(int).tolist()
    log(f"Fully evaluating top {len(selected_for_full)} screened candidates on all {len(frame)} rows")
    rows: list[dict[str, float | str]] = []
    predictions: dict[int, np.ndarray] = {}
    for position, candidate_id in enumerate(selected_for_full, start=1):
        params = candidates[candidate_id]
        metrics, prediction = evaluate_candidate(frame, signals, params, group_infos, targets, candidate_id, args.objective_profile)
        rows.append(metrics)
        predictions[candidate_id] = prediction
        if position % 250 == 0:
            log(f"Fully evaluated {position} screened candidates")

    grid = pd.DataFrame.from_records(rows).sort_values("distribution_score")
    grid.to_csv(run_root / "soft_gate_distribution_grid.csv", index=False)
    selected_ids = grid.head(args.top_k)["candidate_id"].astype(int).tolist()

    log(f"Writing top {len(selected_ids)} candidate scoring artifacts")
    artifact_rows = []
    for rank_idx, candidate_id in enumerate(selected_ids, start=1):
        if candidate_id not in predictions:
            _, predictions[candidate_id] = evaluate_candidate(
                frame,
                signals,
                candidates[candidate_id],
                group_infos,
                targets,
                candidate_id,
                args.objective_profile,
            )
        row = grid.loc[grid["candidate_id"] == candidate_id].iloc[0].to_dict()
        name = f"candidate_{rank_idx:02d}_id_{candidate_id}"
        candidate_dir = write_candidate_artifact(args.scoring_dir, run_root, frame, predictions[candidate_id], row, name)
        artifact_rows.append({"rank": rank_idx, "candidate_id": candidate_id, "candidate_dir": str(candidate_dir), **row})
    pd.DataFrame.from_records(artifact_rows).to_csv(run_root / "selected_candidates.csv", index=False)

    best_dir = run_root / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(run_root / "candidate_01_id_{}".format(selected_ids[0]), best_dir)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_scoring_dir": str(args.scoring_dir),
        "run_root": str(run_root),
        "trials": args.trials,
        "candidate_count": len(candidates),
        "screen_sample_rows": sample_n,
        "screen_top_k": args.screen_top_k,
        "objective_profile": args.objective_profile,
        "best_candidate_id": int(selected_ids[0]),
        "best_dir": str(best_dir),
        "outputs": {
            "screen_grid": str(run_root / "soft_gate_distribution_screen_grid.csv"),
            "grid": str(run_root / "soft_gate_distribution_grid.csv"),
            "selected_candidates": str(run_root / "selected_candidates.csv"),
            "best": str(best_dir),
        },
    }
    (run_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(grid.head(25).to_string(index=False), flush=True)
    log(f"Wrote SIP soft-gate distribution optimization to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
