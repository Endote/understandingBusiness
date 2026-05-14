# Weeklies soldqty time-series analysis

Important: the fact table has no transaction date. The product daily series is synthetic: observed product totals are allocated evenly across the product active interval.

## Active interval convention

`[ONSALEDATE, OFFSALEDATE)` is used, meaning the on-sale date is included and the off-sale date is excluded. This avoids double-counting the date a product is removed from sale.

## Main files

- `product_lifecycle_summary.csv`: one row per product with sale window, observed totals, per-day allocation, and product coherence flag.
- `product_daily_soldqty_timeseries.csv`: one row per active product-date with allocated soldqty/drawqty.
- `calendar_daily_allocated_soldqty.csv`: daily aggregate over all active products.
- `lifecycle_day_allocated_soldqty.csv`: aggregate by lifecycle day since on-sale.
- `coherence_reconstruction_by_product.csv`: verifies the daily series reconstructs product totals.
- `coherence_flag_summary.csv`: counts products by data-quality/coherence flag.
- `product_type_timeseries_summary.csv`: product type-level summary; for a single product group this is the group total.

## Coherence summary

- `analysis_scope`: Weeklies
- `source_limitation`: FACT_TABLE.csv is not dated; daily product soldqty is an even allocation over [ONSALEDATE, OFFSALEDATE). It is suitable for coherence and lifecycle shape checks, not for claiming observed daily demand.
- `interval_convention`: [ONSALEDATE, OFFSALEDATE); if OFFSALEDATE <= ONSALEDATE active_days is invalid and excluded from daily series.
- `product_rows`: 2088
- `fact_products`: 2088
- `fact_rows`: 5700450
- `daily_timeseries_rows`: 32230
- `calendar_start`: 2023-01-02
- `calendar_end`: 2024-07-10
- `missing_dim_product_ids_in_fact_count`: 0
- `missing_dim_product_ids_in_fact_sample`: []
- `invalid_sale_window_products`: 0
- `products_without_fact_rows`: 0
- `products_with_negative_soldqty`: 1219
- `max_raw_reconstruction_abs_error`: 3.637978807091713e-12
- `max_nonnegative_reconstruction_abs_error`: 4.547473508864641e-13
- `max_drawqty_reconstruction_abs_error`: 7.275957614183426e-12

## Fact quality

- `fact_rows`: 5700450
- `fact_products`: 2088
- `total_soldqty_raw`: 8312665
- `total_soldqty_nonnegative`: 8321535
- `total_drawqty`: 46240829
- `negative_soldqty_rows`: 4085
- `zero_soldqty_rows`: 3635793
- `stockout_proxy_rows`: 132656
- `aggregate_sellthrough_raw`: 0.1797689440213107
- `aggregate_sellthrough_nonnegative`: 0.17996076584180618
