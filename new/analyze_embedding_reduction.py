#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import trustworthiness
from sklearn.random_projection import GaussianRandomProjection


NEW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NEW_DIR / "output" / "embedding_analysis"
DEFAULT_DB_URI = os.getenv("PAPERRUSH_DB_URI", "postgresql://norbert.jaworski@/paperrush?host=/tmp")
RANDOM_STATE = 42
QUALITY_DIMS = [2, 5, 10, 16, 32, 64]
UMAP_DIMS = [2, 10, 32]

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def psql_csv(db_uri: str, query: str) -> pd.DataFrame:
    cmd = [
        "psql",
        db_uri,
        "-v",
        "ON_ERROR_STOP=1",
        "-P",
        "pager=off",
        "-c",
        f"COPY ({query}) TO STDOUT WITH CSV HEADER",
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return pd.read_csv(io.StringIO(result.stdout))


def parse_embedding(value: str) -> np.ndarray:
    return np.fromstring(value.strip("[]"), sep=",", dtype=np.float32)


def load_core_embeddings(db_uri: str) -> tuple[pd.DataFrame, np.ndarray]:
    query = """
    select
        dp.product_id,
        dp.type as family,
        coalesce(nullif(trim(dp.title), ''), 'UNKNOWN') as title,
        coalesce(nullif(trim(dp.segment), ''), 'UNKNOWN') as segment,
        coalesce(nullif(trim(dp.subsegment), ''), 'UNKNOWN') as subsegment,
        dp.onsaledate,
        count(fs.*) filter (where fs.drawqty <> 0) as observed_rows,
        count(fs.*) filter (where fs.drawqty <> 0 and fs.soldqty > 0) as positive_rows,
        coalesce(avg(fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as avg_positive_sales,
        coalesce(percentile_cont(0.90) within group (order by fs.soldqty) filter (where fs.drawqty <> 0 and fs.soldqty > 0), 0) as p90_positive_sales,
        ce.embedding::text as embedding
    from core.dim_product dp
    join core.content_embedding ce on ce.product_id = dp.product_id
    left join core.fact_sale fs on fs.product_id = dp.product_id
    group by
        dp.product_id,
        dp.type,
        dp.title,
        dp.segment,
        dp.subsegment,
        dp.onsaledate,
        ce.embedding
    order by dp.type, dp.onsaledate, dp.product_id
    """
    frame = psql_csv(db_uri, query)
    matrix = np.vstack(frame["embedding"].map(parse_embedding).to_numpy()).astype(np.float64)
    frame = frame.drop(columns=["embedding"])
    frame["onsaledate"] = pd.to_datetime(frame["onsaledate"])
    return frame, matrix


def load_holdout_coverage(db_uri: str) -> pd.DataFrame:
    query = """
    select
        dp.type as family,
        count(*) as schedule_rows,
        count(*) filter (where ce.product_id is not null) as embedded_schedule_rows,
        count(distinct ps.product_id) as products,
        count(distinct ps.product_id) filter (where ce.product_id is not null) as embedded_products
    from holdout.printing_schedule ps
    join holdout.dim_product dp on dp.product_id = ps.product_id
    left join holdout.content_embedding ce on ce.product_id = ps.product_id
    group by dp.type
    order by dp.type
    """
    return psql_csv(db_uri, query)


def dims_for_variance(cumulative: np.ndarray, thresholds: list[float]) -> dict[str, int]:
    return {f"dim_for_{int(threshold * 100)}pct_variance": int(np.searchsorted(cumulative, threshold) + 1) for threshold in thresholds}


def pca_variance_table(matrix: np.ndarray, label: str) -> tuple[pd.DataFrame, dict[str, int]]:
    centered = matrix.astype(np.float64)
    pca = PCA(n_components=min(centered.shape), svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(centered)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    rows = []
    for idx, (ratio, total) in enumerate(zip(pca.explained_variance_ratio_, cumulative), start=1):
        rows.append(
            {
                "slice": label,
                "component": idx,
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance": float(total),
            }
        )
    return pd.DataFrame.from_records(rows), dims_for_variance(cumulative, [0.80, 0.85, 0.90, 0.95])


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def method_quality(matrix: np.ndarray, dims: list[int], run_umap: bool) -> pd.DataFrame:
    # Cosine geometry is usually the meaningful geometry for text/content embeddings.
    normalized = row_normalize(matrix.astype(np.float64))
    rows = []
    max_dim = max(dims)

    pca = PCA(n_components=max_dim, random_state=RANDOM_STATE)
    pca_embedding = pca.fit_transform(normalized)
    svd = TruncatedSVD(n_components=max_dim, random_state=RANDOM_STATE)
    svd_embedding = svd.fit_transform(normalized)

    methods = {
        "pca": pca_embedding,
        "truncated_svd": svd_embedding,
    }
    for dim in dims:
        projection = GaussianRandomProjection(n_components=dim, random_state=RANDOM_STATE)
        methods[f"gaussian_random_projection_{dim}"] = projection.fit_transform(normalized)

    if run_umap:
        try:
            import umap

            for dim in [dim for dim in UMAP_DIMS if dim in dims]:
                reducer = umap.UMAP(
                    n_components=dim,
                    n_neighbors=15,
                    min_dist=0.05,
                    metric="cosine",
                    random_state=RANDOM_STATE,
                    low_memory=True,
                )
                methods[f"umap_{dim}"] = reducer.fit_transform(normalized)
        except Exception as exc:
            rows.append({"method": "umap", "dimensions": -1, "trustworthiness_15": np.nan, "error": repr(exc)})

    for name, embedding in methods.items():
        if name in {"pca", "truncated_svd"}:
            dim_candidates = dims
        else:
            dim_candidates = [int(name.rsplit("_", 1)[-1])]
        for dim in dim_candidates:
            if dim > embedding.shape[1]:
                continue
            reduced = embedding[:, :dim]
            rows.append(
                {
                    "method": name if name in {"pca", "truncated_svd"} else name.rsplit("_", 1)[0],
                    "dimensions": dim,
                    "trustworthiness_15": float(trustworthiness(normalized, reduced, n_neighbors=15, metric="cosine")),
                    "error": "",
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze content embedding dimensionality reduction.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-umap", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log("Loading core product embeddings")
    products, matrix = load_core_embeddings(args.db_uri)
    log(f"Loaded {len(products)} products with {matrix.shape[1]} embedding dimensions")

    products.to_csv(args.output_dir / "core_embedding_products.csv", index=False)
    holdout_coverage = load_holdout_coverage(args.db_uri)
    holdout_coverage.to_csv(args.output_dir / "holdout_embedding_coverage.csv", index=False)

    variance_frames = []
    threshold_rows = []
    for label, index in [("all", products.index), *[(family, products.index[products["family"] == family]) for family in sorted(products["family"].unique())]]:
        table, thresholds = pca_variance_table(matrix[index], label)
        variance_frames.append(table)
        threshold_rows.append({"slice": label, "products": int(len(index)), **thresholds})
    variance = pd.concat(variance_frames, ignore_index=True)
    thresholds = pd.DataFrame.from_records(threshold_rows)
    variance.to_csv(args.output_dir / "pca_explained_variance.csv", index=False)
    thresholds.to_csv(args.output_dir / "pca_variance_thresholds.csv", index=False)

    log("Computing reduction quality metrics")
    quality = method_quality(matrix, dims=QUALITY_DIMS, run_umap=args.run_umap)
    quality.to_csv(args.output_dir / "reduction_quality.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "products": int(len(products)),
        "embedding_dimensions": int(matrix.shape[1]),
        "families": products["family"].value_counts().to_dict(),
        "pca_variance_thresholds": thresholds.to_dict(orient="records"),
        "holdout_embedding_coverage": holdout_coverage.to_dict(orient="records"),
        "run_umap": bool(args.run_umap),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("PCA variance thresholds")
    print(thresholds.to_string(index=False), flush=True)
    log("Reduction quality")
    print(quality.sort_values(["method", "dimensions"]).to_string(index=False), flush=True)
    log(f"Wrote embedding analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
