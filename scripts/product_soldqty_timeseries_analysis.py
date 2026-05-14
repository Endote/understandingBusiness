#!/usr/bin/env python3
"""Build product-level sold quantity time-series diagnostics.

The source fact table is not transactional: it contains one historical total per
STORE_ID x PRODUCT_ID. This script therefore creates a synthetic daily product
series by allocating each product's observed total evenly across its active
on-sale interval. Coherence checks verify that the daily series reconstructs the
observed product totals exactly and flag where the source data cannot support a
true dated sales curve.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def make_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "output" / "product_timeseries" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_product(path: Path) -> pd.DataFrame:
    product = normalize_columns(pd.read_csv(path))
    product["onsaledate"] = pd.to_datetime(product["onsaledate"], errors="coerce")
    product["offsaledate"] = pd.to_datetime(product["offsaledate"], errors="coerce")
    product["product_id"] = pd.to_numeric(product["product_id"], errors="coerce").astype("Int64")
    product["price"] = pd.to_numeric(product.get("price"), errors="coerce")
    return product


def aggregate_fact(path: Path, chunksize: int) -> tuple[pd.DataFrame, dict[str, int | float]]:
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    total_soldqty = 0
    total_drawqty = 0
    negative_rows = 0
    zero_rows = 0
    stockout_rows = 0

    usecols = ["STORE_ID", "PRODUCT_ID", "SOLDQTY", "DRAWQTY"]
    dtypes = {
        "STORE_ID": "int64",
        "PRODUCT_ID": "int64",
        "SOLDQTY": "int32",
        "DRAWQTY": "int32",
    }
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        chunk = normalize_columns(chunk)
        chunk["soldqty_nonnegative"] = chunk["soldqty"].clip(lower=0)
        chunk["oversupply_units"] = (chunk["drawqty"] - chunk["soldqty_nonnegative"]).clip(lower=0)
        chunk["negative_soldqty_rows"] = (chunk["soldqty"] < 0).astype("int64")
        chunk["zero_soldqty_rows"] = (chunk["soldqty"] == 0).astype("int64")
        chunk["stockout_proxy_rows"] = ((chunk["drawqty"] > 0) & (chunk["soldqty"] == chunk["drawqty"])).astype("int64")

        total_rows += len(chunk)
        total_soldqty += int(chunk["soldqty"].sum())
        total_drawqty += int(chunk["drawqty"].sum())
        negative_rows += int(chunk["negative_soldqty_rows"].sum())
        zero_rows += int(chunk["zero_soldqty_rows"].sum())
        stockout_rows += int(chunk["stockout_proxy_rows"].sum())

        agg = (
            chunk.groupby("product_id", as_index=False)
            .agg(
                fact_rows=("store_id", "size"),
                soldqty_sum=("soldqty", "sum"),
                soldqty_nonnegative_sum=("soldqty_nonnegative", "sum"),
                drawqty_sum=("drawqty", "sum"),
                oversupply_units=("oversupply_units", "sum"),
                negative_soldqty_rows=("negative_soldqty_rows", "sum"),
                zero_soldqty_rows=("zero_soldqty_rows", "sum"),
                stockout_proxy_rows=("stockout_proxy_rows", "sum"),
            )
        )
        chunks.append(agg)

    fact_product = (
        pd.concat(chunks, ignore_index=True)
        .groupby("product_id", as_index=False)
        .sum(numeric_only=True)
    )
    fact_product["aggregate_sellthrough_raw"] = np.where(
        fact_product["drawqty_sum"] > 0,
        fact_product["soldqty_sum"] / fact_product["drawqty_sum"],
        np.nan,
    )
    fact_product["aggregate_sellthrough_nonnegative"] = np.where(
        fact_product["drawqty_sum"] > 0,
        fact_product["soldqty_nonnegative_sum"] / fact_product["drawqty_sum"],
        np.nan,
    )

    fact_quality = {
        "fact_rows": total_rows,
        "fact_products": int(fact_product["product_id"].nunique()),
        "total_soldqty_raw": total_soldqty,
        "total_drawqty": total_drawqty,
        "negative_soldqty_rows": negative_rows,
        "zero_soldqty_rows": zero_rows,
        "stockout_proxy_rows": stockout_rows,
        "aggregate_sellthrough_raw": total_soldqty / total_drawqty if total_drawqty else np.nan,
    }
    return fact_product, fact_quality


def build_product_summary(product: pd.DataFrame, fact_product: pd.DataFrame) -> pd.DataFrame:
    summary = product.merge(fact_product, on="product_id", how="left", indicator=True)
    numeric_fill = [
        "fact_rows",
        "soldqty_sum",
        "soldqty_nonnegative_sum",
        "drawqty_sum",
        "oversupply_units",
        "negative_soldqty_rows",
        "zero_soldqty_rows",
        "stockout_proxy_rows",
    ]
    for col in numeric_fill:
        summary[col] = summary[col].fillna(0)

    summary["has_fact_rows"] = summary["_merge"].eq("both")
    summary = summary.drop(columns=["_merge"])
    summary["active_days_exclusive"] = (summary["offsaledate"] - summary["onsaledate"]).dt.days
    summary["active_days_inclusive"] = summary["active_days_exclusive"] + 1
    summary["active_days"] = summary["active_days_exclusive"].clip(lower=1)
    summary["invalid_sale_window"] = (
        summary["onsaledate"].isna()
        | summary["offsaledate"].isna()
        | (summary["offsaledate"] <= summary["onsaledate"])
    )
    summary["daily_soldqty_raw_alloc"] = np.where(
        summary["active_days"] > 0,
        summary["soldqty_sum"] / summary["active_days"],
        np.nan,
    )
    summary["daily_soldqty_nonnegative_alloc"] = np.where(
        summary["active_days"] > 0,
        summary["soldqty_nonnegative_sum"] / summary["active_days"],
        np.nan,
    )
    summary["daily_drawqty_alloc"] = np.where(
        summary["active_days"] > 0,
        summary["drawqty_sum"] / summary["active_days"],
        np.nan,
    )
    summary["product_coherence_flag"] = np.select(
        [
            summary["invalid_sale_window"],
            ~summary["has_fact_rows"],
            summary["negative_soldqty_rows"] > 0,
            summary["drawqty_sum"].eq(0) & summary["soldqty_sum"].ne(0),
        ],
        [
            "invalid_sale_window",
            "missing_fact_rows",
            "has_negative_soldqty",
            "sold_without_draw",
        ],
        default="ok",
    )
    return summary


def build_daily_series(summary: pd.DataFrame) -> pd.DataFrame:
    valid = summary.loc[~summary["invalid_sale_window"]].copy()
    pieces: list[pd.DataFrame] = []
    base_cols = [
        "product_id",
        "barcode",
        "title",
        "type",
        "segment",
        "subsegment",
        "frequency",
        "onsaledate",
        "offsaledate",
        "active_days",
        "fact_rows",
        "soldqty_sum",
        "soldqty_nonnegative_sum",
        "drawqty_sum",
        "daily_soldqty_raw_alloc",
        "daily_soldqty_nonnegative_alloc",
        "daily_drawqty_alloc",
        "product_coherence_flag",
    ]
    for row in valid[base_cols].itertuples(index=False):
        dates = pd.date_range(row.onsaledate, periods=int(row.active_days), freq="D")
        frame = pd.DataFrame(
            {
                "calendar_date": dates,
                "lifecycle_day": np.arange(1, len(dates) + 1, dtype="int32"),
                "lifecycle_pct": np.arange(1, len(dates) + 1, dtype="float64") / len(dates),
            }
        )
        for col, value in zip(base_cols, row):
            frame[col] = value
        pieces.append(frame)

    daily = pd.concat(pieces, ignore_index=True)
    ordered = [
        "product_id",
        "calendar_date",
        "lifecycle_day",
        "lifecycle_pct",
        "daily_soldqty_raw_alloc",
        "daily_soldqty_nonnegative_alloc",
        "daily_drawqty_alloc",
        "soldqty_sum",
        "soldqty_nonnegative_sum",
        "drawqty_sum",
        "fact_rows",
        "active_days",
        "onsaledate",
        "offsaledate",
        "title",
        "type",
        "segment",
        "subsegment",
        "frequency",
        "barcode",
        "product_coherence_flag",
    ]
    return daily[ordered]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def save_plots(output_dir: Path, summary: pd.DataFrame, daily: pd.DataFrame, label: str) -> None:
    sns.set_theme(style="whitegrid")

    by_date = (
        daily.groupby("calendar_date", as_index=False)
        .agg(
            active_products=("product_id", "nunique"),
            soldqty_daily_alloc=("daily_soldqty_nonnegative_alloc", "sum"),
            drawqty_daily_alloc=("daily_drawqty_alloc", "sum"),
        )
    )
    by_date["allocated_sellthrough"] = by_date["soldqty_daily_alloc"] / by_date["drawqty_daily_alloc"]
    write_csv(by_date, output_dir / "calendar_daily_allocated_soldqty.csv")

    by_lifecycle = (
        daily.groupby("lifecycle_day", as_index=False)
        .agg(
            active_products=("product_id", "nunique"),
            soldqty_daily_alloc=("daily_soldqty_nonnegative_alloc", "sum"),
            drawqty_daily_alloc=("daily_drawqty_alloc", "sum"),
        )
    )
    by_lifecycle["allocated_sellthrough"] = by_lifecycle["soldqty_daily_alloc"] / by_lifecycle["drawqty_daily_alloc"]
    write_csv(by_lifecycle, output_dir / "lifecycle_day_allocated_soldqty.csv")

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=by_date, x="calendar_date", y="soldqty_daily_alloc", color="#1f77b4")
    plt.title(f"{label}: allocated daily sold quantity across active products")
    plt.xlabel("Calendar date")
    plt.ylabel("Allocated soldqty")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_calendar_allocated_soldqty.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=by_date, x="calendar_date", y="active_products", color="#2ca02c")
    plt.title(f"{label}: active products by calendar date")
    plt.xlabel("Calendar date")
    plt.ylabel("Active products")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_calendar_active_products.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=by_lifecycle, x="lifecycle_day", y="soldqty_daily_alloc", color="#9467bd")
    plt.title(f"{label}: allocated sold quantity by product lifecycle day")
    plt.xlabel("Lifecycle day")
    plt.ylabel("Allocated soldqty")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_lifecycle_allocated_soldqty.png", dpi=160)
    plt.close()

    month_segment = summary.copy()
    month_segment["onsale_month"] = month_segment["onsaledate"].dt.to_period("M").astype(str)
    segment_col = "type" if month_segment["type"].nunique(dropna=True) > 1 else "segment"
    month_segment = (
        month_segment.groupby(["onsale_month", segment_col], dropna=False, as_index=False)
        .agg(products=("product_id", "nunique"), soldqty=("soldqty_nonnegative_sum", "sum"))
    )
    pivot = month_segment.pivot(index="onsale_month", columns=segment_col, values="soldqty").fillna(0)
    if pivot.shape[1] > 0:
        top_columns = pivot.sum(axis=0).sort_values(ascending=False).head(20).index
        pivot = pivot.loc[:, top_columns]
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, cmap="viridis", linewidths=0.2)
        plt.title(f"{label}: nonnegative soldqty by on-sale month and {segment_col}")
        plt.xlabel(segment_col)
        plt.ylabel("On-sale month")
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_onsale_month_{segment_col}_soldqty_heatmap.png", dpi=160)
        plt.close()

    top = summary.sort_values("soldqty_nonnegative_sum", ascending=False).head(30)
    plt.figure(figsize=(14, 10))
    hue = "type" if top["type"].nunique(dropna=True) > 1 else "segment"
    sns.barplot(data=top, y=top["product_id"].astype(str), x="soldqty_nonnegative_sum", hue=hue, dodge=False)
    plt.title(f"{label}: top 30 products by observed nonnegative soldqty")
    plt.xlabel("Observed soldqty")
    plt.ylabel("Product ID")
    plt.legend(title="Type", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_top_products_soldqty.png", dpi=160)
    plt.close()


def build_fact_quality(summary: pd.DataFrame) -> dict[str, int | float]:
    total_soldqty = float(summary["soldqty_sum"].sum())
    total_drawqty = float(summary["drawqty_sum"].sum())
    return {
        "fact_rows": int(summary["fact_rows"].sum()),
        "fact_products": int(summary.loc[summary["has_fact_rows"], "product_id"].nunique()),
        "total_soldqty_raw": int(total_soldqty),
        "total_soldqty_nonnegative": int(summary["soldqty_nonnegative_sum"].sum()),
        "total_drawqty": int(total_drawqty),
        "negative_soldqty_rows": int(summary["negative_soldqty_rows"].sum()),
        "zero_soldqty_rows": int(summary["zero_soldqty_rows"].sum()),
        "stockout_proxy_rows": int(summary["stockout_proxy_rows"].sum()),
        "aggregate_sellthrough_raw": total_soldqty / total_drawqty if total_drawqty else np.nan,
        "aggregate_sellthrough_nonnegative": (
            float(summary["soldqty_nonnegative_sum"].sum()) / total_drawqty if total_drawqty else np.nan
        ),
    }


def coherence_tables(output_dir: Path, product: pd.DataFrame, fact_product: pd.DataFrame, summary: pd.DataFrame, daily: pd.DataFrame, fact_quality: dict[str, int | float], label: str) -> dict[str, object]:
    daily_recon = (
        daily.groupby("product_id", as_index=False)
        .agg(
            reconstructed_soldqty_raw=("daily_soldqty_raw_alloc", "sum"),
            reconstructed_soldqty_nonnegative=("daily_soldqty_nonnegative_alloc", "sum"),
            reconstructed_drawqty=("daily_drawqty_alloc", "sum"),
            timeseries_rows=("calendar_date", "size"),
        )
    )
    recon = summary.merge(daily_recon, on="product_id", how="left")
    recon["raw_reconstruction_abs_error"] = (recon["reconstructed_soldqty_raw"] - recon["soldqty_sum"]).abs()
    recon["nonnegative_reconstruction_abs_error"] = (
        recon["reconstructed_soldqty_nonnegative"] - recon["soldqty_nonnegative_sum"]
    ).abs()
    recon["drawqty_reconstruction_abs_error"] = (recon["reconstructed_drawqty"] - recon["drawqty_sum"]).abs()
    recon_cols = [
        "product_id",
        "type",
        "title",
        "onsaledate",
        "offsaledate",
        "active_days",
        "fact_rows",
        "soldqty_sum",
        "soldqty_nonnegative_sum",
        "drawqty_sum",
        "timeseries_rows",
        "raw_reconstruction_abs_error",
        "nonnegative_reconstruction_abs_error",
        "drawqty_reconstruction_abs_error",
        "product_coherence_flag",
    ]
    write_csv(recon[recon_cols], output_dir / "coherence_reconstruction_by_product.csv")

    missing_dim_products = sorted(set(fact_product["product_id"]) - set(product["product_id"]))
    flag_summary = (
        summary.groupby("product_coherence_flag", as_index=False)
        .agg(products=("product_id", "nunique"), soldqty_nonnegative_sum=("soldqty_nonnegative_sum", "sum"))
        .sort_values(["products", "soldqty_nonnegative_sum"], ascending=False)
    )
    write_csv(flag_summary, output_dir / "coherence_flag_summary.csv")

    type_summary = (
        summary.groupby("type", dropna=False, as_index=False)
        .agg(
            products=("product_id", "nunique"),
            products_with_fact=("has_fact_rows", "sum"),
            observed_soldqty_raw=("soldqty_sum", "sum"),
            observed_soldqty_nonnegative=("soldqty_nonnegative_sum", "sum"),
            drawqty=("drawqty_sum", "sum"),
            median_active_days=("active_days", "median"),
            negative_soldqty_rows=("negative_soldqty_rows", "sum"),
            stockout_proxy_rows=("stockout_proxy_rows", "sum"),
        )
    )
    type_summary["aggregate_sellthrough_nonnegative"] = type_summary["observed_soldqty_nonnegative"] / type_summary["drawqty"]
    write_csv(type_summary, output_dir / "product_type_timeseries_summary.csv")

    return {
        "analysis_scope": label,
        "source_limitation": "FACT_TABLE.csv is not dated; daily product soldqty is an even allocation over [ONSALEDATE, OFFSALEDATE). It is suitable for coherence and lifecycle shape checks, not for claiming observed daily demand.",
        "interval_convention": "[ONSALEDATE, OFFSALEDATE); if OFFSALEDATE <= ONSALEDATE active_days is invalid and excluded from daily series.",
        "product_rows": int(len(product)),
        "fact_products": int(fact_quality["fact_products"]),
        "fact_rows": int(fact_quality["fact_rows"]),
        "daily_timeseries_rows": int(len(daily)),
        "calendar_start": str(daily["calendar_date"].min().date()),
        "calendar_end": str(daily["calendar_date"].max().date()),
        "missing_dim_product_ids_in_fact_count": len(missing_dim_products),
        "missing_dim_product_ids_in_fact_sample": missing_dim_products[:20],
        "invalid_sale_window_products": int(summary["invalid_sale_window"].sum()),
        "products_without_fact_rows": int((~summary["has_fact_rows"]).sum()),
        "products_with_negative_soldqty": int((summary["negative_soldqty_rows"] > 0).sum()),
        "max_raw_reconstruction_abs_error": float(recon["raw_reconstruction_abs_error"].max(skipna=True)),
        "max_nonnegative_reconstruction_abs_error": float(recon["nonnegative_reconstruction_abs_error"].max(skipna=True)),
        "max_drawqty_reconstruction_abs_error": float(recon["drawqty_reconstruction_abs_error"].max(skipna=True)),
        "fact_quality": fact_quality,
    }


def write_readme(output_dir: Path, summary: dict[str, object]) -> None:
    scope = summary.get("analysis_scope", "Product")
    lines = [
        f"# {scope} soldqty time-series analysis",
        "",
        "Important: the fact table has no transaction date. The product daily series is synthetic: observed product totals are allocated evenly across the product active interval.",
        "",
        "## Active interval convention",
        "",
        "`[ONSALEDATE, OFFSALEDATE)` is used, meaning the on-sale date is included and the off-sale date is excluded. This avoids double-counting the date a product is removed from sale.",
        "",
        "## Main files",
        "",
        "- `product_lifecycle_summary.csv`: one row per product with sale window, observed totals, per-day allocation, and product coherence flag.",
        "- `product_daily_soldqty_timeseries.csv`: one row per active product-date with allocated soldqty/drawqty.",
        "- `calendar_daily_allocated_soldqty.csv`: daily aggregate over all active products.",
        "- `lifecycle_day_allocated_soldqty.csv`: aggregate by lifecycle day since on-sale.",
        "- `coherence_reconstruction_by_product.csv`: verifies the daily series reconstructs product totals.",
        "- `coherence_flag_summary.csv`: counts products by data-quality/coherence flag.",
        "- `product_type_timeseries_summary.csv`: product type-level summary; for a single product group this is the group total.",
        "",
        "## Coherence summary",
        "",
    ]
    for key, value in summary.items():
        if key == "fact_quality":
            continue
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Fact quality", ""])
    for key, value in summary["fact_quality"].items():
        lines.append(f"- `{key}`: {value}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_bundle(
    output_dir: Path,
    product: pd.DataFrame,
    fact_product: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    label: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, output_dir / "product_lifecycle_summary.csv")
    write_csv(daily, output_dir / "product_daily_soldqty_timeseries.csv")
    save_plots(output_dir, summary, daily, label)
    run_summary = coherence_tables(output_dir, product, fact_product, summary, daily, build_fact_quality(summary), label)
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")
    write_readme(output_dir, run_summary)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=Path, default=INPUT_DIR / "DIM_PRODUCT.csv")
    parser.add_argument("--fact", type=Path, default=INPUT_DIR / "FACT_TABLE.csv")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    output_dir = make_output_dir(ROOT)
    print(f"Writing outputs to {output_dir}")
    product = load_product(args.product)
    fact_product, fact_quality = aggregate_fact(args.fact, args.chunksize)
    summary = build_product_summary(product, fact_product)
    daily = build_daily_series(summary)

    run_summary = write_analysis_bundle(output_dir, product, fact_product, summary, daily, "All products")

    group_summaries = {}
    for product_type in ["Weeklies", "SIP"]:
        group_slug = safe_slug(product_type)
        group_summary = summary.loc[summary["type"].eq(product_type)].copy()
        group_daily = daily.loc[daily["type"].eq(product_type)].copy()
        group_product = product.loc[product["type"].eq(product_type)].copy()
        group_fact = fact_product.loc[fact_product["product_id"].isin(group_summary["product_id"])].copy()
        group_dir = output_dir / group_slug
        group_summaries[group_slug] = write_analysis_bundle(
            group_dir,
            group_product,
            group_fact,
            group_summary,
            group_daily,
            product_type,
        )

    grouped_index = pd.DataFrame(
        [
            {
                "product_group": value["analysis_scope"],
                "folder": slug,
                "product_rows": value["product_rows"],
                "daily_timeseries_rows": value["daily_timeseries_rows"],
                "calendar_start": value["calendar_start"],
                "calendar_end": value["calendar_end"],
                "products_with_negative_soldqty": value["products_with_negative_soldqty"],
                "aggregate_sellthrough_nonnegative": value["fact_quality"]["aggregate_sellthrough_nonnegative"],
            }
            for slug, value in group_summaries.items()
        ]
    )
    write_csv(grouped_index, output_dir / "grouped_analysis_index.csv")

    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "all_products": run_summary,
                "groups": group_summaries,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
