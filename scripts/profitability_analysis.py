#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URI = os.getenv("PAPERRUSH_DB_URI", "postgresql://norbert.jaworski@/paperrush?host=/tmp")
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "output" / "profitability_analysis"

RETAILER_MARGIN_RATE = 0.15
PUBLISHER_REVENUE_RATE = 1.0 - RETAILER_MARGIN_RATE
REVENUE_TO_PRODUCTION_COST_RATIO = 5.0
PRODUCTION_COST_RATE = PUBLISHER_REVENUE_RATE / REVENUE_TO_PRODUCTION_COST_RATIO
BREAKEVEN_SELLTHROUGH = PRODUCTION_COST_RATE / PUBLISHER_REVENUE_RATE


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_psql_csv(db_uri: str, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_cmd = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
    cmd = ["psql", db_uri, "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-c", copy_cmd]
    log("RUN: " + " ".join(cmd[:-1]) + " \"COPY (...) TO STDOUT WITH CSV HEADER\"")
    with destination.open("wb") as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def run_psql_script(db_uri: str, script: str) -> None:
    cmd = ["psql", db_uri, "-v", "ON_ERROR_STOP=1", "-P", "pager=off"]
    log("RUN: " + " ".join(cmd) + " < generated profitability export SQL")
    result = subprocess.run(cmd, input=script, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def psql_literal(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def one_line_sql(query: str) -> str:
    return " ".join(query.split())


def money_expr(sold_expr: str = "fs.soldqty") -> str:
    return (
        f"(({sold_expr}) * dp.price * {PUBLISHER_REVENUE_RATE}) "
        f"- (fs.drawqty * dp.price * {PRODUCTION_COST_RATE})"
    )


def base_from() -> str:
    return """
        core.fact_sale fs
        join core.dim_product dp on dp.product_id = fs.product_id
        join core.dim_store ds on ds.store_id = fs.store_id
    """


def metric_select(group_cols: list[str]) -> str:
    group_prefix = ",\n        ".join(group_cols)
    group_clause = ", ".join(str(idx + 1) for idx in range(len(group_cols)))
    return f"""
    select
        {group_prefix},
        count(*) as sale_rows,
        count(distinct fs.product_id) as products,
        count(distinct fs.store_id) as stores,
        sum(fs.drawqty) as draw_qty,
        sum(fs.soldqty) as raw_sold_qty,
        sum(greatest(fs.soldqty, 0)) as demand_sold_qty,
        round(sum(fs.soldqty)::numeric / nullif(sum(fs.drawqty), 0), 6) as accounting_sellthrough,
        round(sum(greatest(fs.soldqty, 0))::numeric / nullif(sum(fs.drawqty), 0), 6) as demand_sellthrough,
        round(sum(fs.soldqty * dp.price * {PUBLISHER_REVENUE_RATE})::numeric, 2) as accounting_publisher_revenue,
        round(sum(greatest(fs.soldqty, 0) * dp.price * {PUBLISHER_REVENUE_RATE})::numeric, 2) as demand_publisher_revenue,
        round(sum(fs.drawqty * dp.price * {PRODUCTION_COST_RATE})::numeric, 2) as production_cost,
        round(sum({money_expr("fs.soldqty")})::numeric, 2) as accounting_gross_profit,
        round(sum({money_expr("greatest(fs.soldqty, 0)")})::numeric, 2) as demand_view_gross_profit,
        round(sum({money_expr("fs.soldqty")})::numeric / nullif(sum(fs.soldqty * dp.price * {PUBLISHER_REVENUE_RATE}), 0), 6) as accounting_profit_margin_on_revenue,
        round(sum({money_expr("greatest(fs.soldqty, 0)")})::numeric / nullif(sum(greatest(fs.soldqty, 0) * dp.price * {PUBLISHER_REVENUE_RATE}), 0), 6) as demand_profit_margin_on_revenue
    from {base_from()}
    group by {group_clause}
    """


def temp_metric_select(group_cols: list[str]) -> str:
    group_prefix = ",\n        ".join(group_cols)
    group_clause = ", ".join(str(idx + 1) for idx in range(len(group_cols)))
    return f"""
    select
        {group_prefix},
        count(*) as sale_rows,
        count(distinct product_id) as products,
        count(distinct store_id) as stores,
        sum(drawqty) as draw_qty,
        sum(raw_sold_qty) as raw_sold_qty,
        sum(demand_sold_qty) as demand_sold_qty,
        round(sum(raw_sold_qty)::numeric / nullif(sum(drawqty), 0), 6) as accounting_sellthrough,
        round(sum(demand_sold_qty)::numeric / nullif(sum(drawqty), 0), 6) as demand_sellthrough,
        round(sum(accounting_publisher_revenue)::numeric, 2) as accounting_publisher_revenue,
        round(sum(demand_publisher_revenue)::numeric, 2) as demand_publisher_revenue,
        round(sum(production_cost)::numeric, 2) as production_cost,
        round(sum(accounting_gross_profit)::numeric, 2) as accounting_gross_profit,
        round(sum(demand_view_gross_profit)::numeric, 2) as demand_view_gross_profit,
        round(sum(accounting_gross_profit)::numeric / nullif(sum(accounting_publisher_revenue), 0), 6) as accounting_profit_margin_on_revenue,
        round(sum(demand_view_gross_profit)::numeric / nullif(sum(demand_publisher_revenue), 0), 6) as demand_profit_margin_on_revenue
    from profitability_line_items
    group by {group_clause}
    """


def ordered_metric_query(group_cols: list[str], order_by: str) -> str:
    return f"select * from ({metric_select(group_cols)}) grouped order by {order_by}"


def ordered_temp_metric_query(group_cols: list[str], order_by: str) -> str:
    return f"select * from ({temp_metric_select(group_cols)}) grouped order by {order_by}"


def write_assumptions(output_dir: Path, db_uri: str) -> None:
    assumptions = {
        "database_uri": db_uri,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Kearney - PaperRush Business Case PDF assumptions quoted by user",
        "retailer_margin_rate_of_cover_price": RETAILER_MARGIN_RATE,
        "publisher_revenue_rate_of_cover_price": PUBLISHER_REVENUE_RATE,
        "publisher_revenue_to_production_cost_ratio": REVENUE_TO_PRODUCTION_COST_RATIO,
        "production_cost_rate_of_cover_price": PRODUCTION_COST_RATE,
        "breakeven_sellthrough": BREAKEVEN_SELLTHROUGH,
        "accounting_profit_formula": "soldqty * price * 0.85 - drawqty * price * 0.17",
        "demand_view_profit_formula": "greatest(soldqty, 0) * price * 0.85 - drawqty * price * 0.17",
        "modeling_target_note": "Use greatest(soldqty, 0) for demand forecasting; keep raw soldqty for accounting cuts because negative rows may be returns or corrections.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "financial_assumptions.json").write_text(
        json.dumps(assumptions, indent=2) + "\n",
        encoding="utf-8",
    )


def export_reports(db_uri: str, output_dir: Path) -> None:
    write_assumptions(output_dir, db_uri)

    reports = {
        "current_summary.csv": """
            select
                min(onsaledate) as min_onsaledate,
                max(onsaledate) as max_onsaledate,
                count(*) as sale_rows,
                count(distinct product_id) as products,
                count(distinct store_id) as stores,
                sum(case when raw_sold_qty < 0 then 1 else 0 end) as negative_sale_rows,
                sum(case when raw_sold_qty < 0 then raw_sold_qty else 0 end) as negative_sold_qty,
                sum(drawqty) as draw_qty,
                sum(raw_sold_qty) as raw_sold_qty,
                sum(demand_sold_qty) as demand_sold_qty,
                round(sum(raw_sold_qty)::numeric / nullif(sum(drawqty), 0), 6) as accounting_sellthrough,
                round(sum(demand_sold_qty)::numeric / nullif(sum(drawqty), 0), 6) as demand_sellthrough,
                round(sum(accounting_publisher_revenue)::numeric, 2) as accounting_publisher_revenue,
                round(sum(demand_publisher_revenue)::numeric, 2) as demand_publisher_revenue,
                round(sum(production_cost)::numeric, 2) as production_cost,
                round(sum(accounting_gross_profit)::numeric, 2) as accounting_gross_profit,
                round(sum(demand_view_gross_profit)::numeric, 2) as demand_view_gross_profit
            from profitability_line_items
        """,
        "by_year.csv": ordered_temp_metric_query(
            ["extract(year from onsaledate)::int as year"],
            "year",
        ),
        "by_month.csv": ordered_temp_metric_query(
            ["to_char(date_trunc('month', onsaledate), 'YYYY-MM') as month"],
            "month",
        ),
        "by_type.csv": ordered_temp_metric_query(["type"], "accounting_gross_profit"),
        "by_store_chain.csv": ordered_temp_metric_query(["store_chain"], "accounting_gross_profit"),
        "by_classoftrade.csv": ordered_temp_metric_query(["classoftrade"], "accounting_gross_profit"),
        "by_region.csv": ordered_temp_metric_query(["region"], "accounting_gross_profit"),
        "by_title.csv": ordered_temp_metric_query(["title", "type"], "accounting_gross_profit"),
        "oracle_same_sales_upper_bound_by_type.csv": """
            select
                type,
                count(distinct product_id) as products,
                count(distinct store_id) as stores,
                sum(drawqty) as current_draw_qty,
                sum(demand_sold_qty) as observed_nonnegative_sales_qty,
                round(sum(demand_view_gross_profit)::numeric, 2) as current_demand_view_gross_profit,
                round(sum((demand_sold_qty * price * 0.85) - (demand_sold_qty * price * 0.17))::numeric, 2) as oracle_same_sales_gross_profit,
                round(sum((drawqty - demand_sold_qty) * price * 0.17)::numeric, 2) as avoidable_production_cost_upper_bound
            from profitability_line_items
            group by 1
            order by avoidable_production_cost_upper_bound desc
        """,
        "holdout_schedule_scope.csv": """
            select
                count(*) as schedule_rows,
                count(distinct ps.product_id) as products,
                count(distinct ps.store_id) as stores,
                min(dp.onsaledate) as min_onsaledate,
                max(dp.onsaledate) as max_onsaledate
            from holdout.printing_schedule ps
            join holdout.dim_product dp on dp.product_id = ps.product_id
        """,
    }

    export_sql = [
        "create temp table profitability_line_items as",
        f"""
        select
            fs.product_id,
            fs.store_id,
            dp.onsaledate,
            dp.price::double precision as price,
            dp.type,
            dp.title,
            ds.store_chain,
            ds.classoftrade,
            ds.region,
            fs.drawqty,
            fs.soldqty as raw_sold_qty,
            greatest(fs.soldqty, 0) as demand_sold_qty,
            fs.soldqty * dp.price * {PUBLISHER_REVENUE_RATE} as accounting_publisher_revenue,
            greatest(fs.soldqty, 0) * dp.price * {PUBLISHER_REVENUE_RATE} as demand_publisher_revenue,
            fs.drawqty * dp.price * {PRODUCTION_COST_RATE} as production_cost,
            {money_expr("fs.soldqty")} as accounting_gross_profit,
            {money_expr("greatest(fs.soldqty, 0)")} as demand_view_gross_profit
        from {base_from()};
        """,
        "analyze profitability_line_items;",
    ]

    for file_name, query in reports.items():
        destination = output_dir / file_name
        log(f"Queueing {destination}")
        export_sql.append(f"\\copy ({one_line_sql(query)}) TO {psql_literal(destination)} WITH CSV HEADER")

    run_psql_script(db_uri, "\n".join(export_sql))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PaperRush profitability analysis from the live database.")
    parser.add_argument("--db-uri", default=DEFAULT_DB_URI)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    export_reports(args.db_uri, output_dir)
    log(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
