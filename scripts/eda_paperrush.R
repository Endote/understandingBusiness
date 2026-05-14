#!/usr/bin/env Rscript


suppressWarnings({
  options(stringsAsFactors = FALSE)
})

required_packages <- c("DBI", "RPostgres")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required R packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them first, e.g. install.packages(c(",
      paste(sprintf("\"%s\"", missing_packages), collapse = ", "),
      "))"
    )
  )
}

library(DBI)
library(RPostgres)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x)) y else x
}

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(sub("^--file=", "", file_arg[1]))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(sys.frames()[[1]]$ofile)
  }
  return(NULL)
}

script_path <- get_script_path()
if (is.null(script_path)) {
  root_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
} else {
  root_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
}

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_dir <- file.path(root_dir, "output", "eda", timestamp)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

log_line <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste(..., collapse = ""))
  cat(msg, "\n")
}

write_csv_safe <- function(df, path) {
  utils::write.csv(df, path, row.names = FALSE, na = "")
}

save_barplot <- function(values, labels, main, ylab, path, las = 2, cex.names = 0.8) {
  png(path, width = 1600, height = 900, res = 140)
  par(mar = c(10, 5, 4, 2))
  barplot(
    values,
    names.arg = labels,
    las = las,
    cex.names = cex.names,
    main = main,
    ylab = ylab,
    col = "steelblue"
  )
  dev.off()
}

save_heatmap <- function(mat, path, main) {
  png(path, width = 1600, height = 1400, res = 140)
  par(mar = c(10, 10, 4, 2))
  heatmap(
    mat,
    Rowv = NA,
    Colv = NA,
    scale = "none",
    col = colorRampPalette(c("#b2182b", "#f7f7f7", "#2166ac"))(100),
    margins = c(10, 10),
    main = main
  )
  dev.off()
}

db_name <- Sys.getenv("PAPERRUSH_DBNAME", "paperrush")
db_host <- Sys.getenv("PAPERRUSH_HOST", "")
db_port <- Sys.getenv("PAPERRUSH_PORT", "")
db_user <- Sys.getenv("PAPERRUSH_USER", Sys.getenv("PGUSER", ""))
db_password <- Sys.getenv("PAPERRUSH_PASSWORD", Sys.getenv("PGPASSWORD", ""))

con_args <- list(
  drv = RPostgres::Postgres(),
  dbname = db_name
)

if (nzchar(db_host)) con_args$host <- db_host
if (nzchar(db_port)) con_args$port <- as.integer(db_port)
if (nzchar(db_user)) con_args$user <- db_user
if (nzchar(db_password)) con_args$password <- db_password

log_line("Connecting to database ", db_name)
con <- do.call(DBI::dbConnect, con_args)
on.exit(DBI::dbDisconnect(con), add = TRUE)

queries <- list(
  table_count = "
    select 'core.fact_sale' as table_name, count(*)::bigint as row_count from core.fact_sale
    union all select 'core.dim_product', count(*)::bigint from core.dim_product
    union all select 'core.dim_store', count(*)::bigint from core.dim_store
    union all select 'core.demographic', count(*)::bigint from core.demographic
    union all select 'core.content_embedding', count(*)::bigint from core.content_embedding
    union all select 'holdout.printing_schedule', count(*)::bigint from holdout.printing_schedule
    union all select 'holdout.dim_product', count(*)::bigint from holdout.dim_product
    union all select 'holdout.dim_store', count(*)::bigint from holdout.dim_store
    union all select 'holdout.demographic', count(*)::bigint from holdout.demographic
    union all select 'holdout.content_embedding', count(*)::bigint from holdout.content_embedding
  ",
  fact_quality = "
    select
      count(*)::bigint as row_count,
      sum(case when soldqty < 0 then 1 else 0 end)::bigint as negative_soldqty_rows,
      sum(case when soldqty = 0 then 1 else 0 end)::bigint as zero_soldqty_rows,
      sum(case when drawqty = 0 then 1 else 0 end)::bigint as zero_drawqty_rows,
      sum(case when drawqty > 0 and soldqty = drawqty then 1 else 0 end)::bigint as stockout_proxy_rows,
      sum(soldqty)::bigint as total_soldqty,
      sum(drawqty)::bigint as total_drawqty,
      avg(case when drawqty > 0 then soldqty::double precision / drawqty else null end) as avg_row_sellthrough
    from core.fact_sale
  ",
  type_summary = "
    select
      dp.type,
      count(*)::bigint as row_count,
      count(distinct fs.store_id)::bigint as store_count,
      count(distinct fs.product_id)::bigint as product_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough,
      avg(case when fs.soldqty = 0 then 1.0 else 0.0 end) as zero_sales_rate,
      avg(case when fs.drawqty > 0 and fs.soldqty = fs.drawqty then 1.0 else 0.0 end) as stockout_proxy_rate
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    group by dp.type
    order by soldqty_sum desc
  ",
  chain_summary = "
    select
      ds.store_chain,
      count(*)::bigint as row_count,
      count(distinct fs.store_id)::bigint as store_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough,
      avg(case when fs.soldqty = 0 then 1.0 else 0.0 end) as zero_sales_rate
    from core.fact_sale fs
    join core.dim_store ds on ds.store_id = fs.store_id
    group by ds.store_chain
    order by soldqty_sum desc
  ",
  class_summary = "
    select
      ds.classoftrade,
      count(*)::bigint as row_count,
      count(distinct fs.store_id)::bigint as store_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough,
      avg(case when fs.soldqty = 0 then 1.0 else 0.0 end) as zero_sales_rate
    from core.fact_sale fs
    join core.dim_store ds on ds.store_id = fs.store_id
    group by ds.classoftrade
    order by soldqty_sum desc
  ",
  region_summary = "
    select
      ds.region,
      count(*)::bigint as row_count,
      count(distinct fs.store_id)::bigint as store_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough,
      avg(case when fs.soldqty = 0 then 1.0 else 0.0 end) as zero_sales_rate
    from core.fact_sale fs
    join core.dim_store ds on ds.store_id = fs.store_id
    group by ds.region
    order by soldqty_sum desc
  ",
  store_mix_summary = "
    select
      fs.store_id,
      ds.store_chain,
      ds.classoftrade,
      ds.region,
      count(distinct fs.product_id)::bigint as product_count,
      count(distinct dp.title)::bigint as title_count,
      count(distinct dp.segment)::bigint as segment_count,
      count(distinct dp.subsegment)::bigint as subsegment_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough
    from core.fact_sale fs
    join core.dim_store ds on ds.store_id = fs.store_id
    join core.dim_product dp on dp.product_id = fs.product_id
    group by fs.store_id, ds.store_chain, ds.classoftrade, ds.region
    order by soldqty_sum desc
  ",
  weeklies_title_summary = "
    select
      dp.title,
      dp.onsaledate,
      count(distinct fs.store_id)::bigint as store_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    where dp.type = 'Weeklies'
    group by dp.title, dp.onsaledate
    order by dp.title, dp.onsaledate
  ",
  sip_segment_summary = "
    select
      dp.segment,
      dp.subsegment,
      count(*)::bigint as row_count,
      count(distinct fs.store_id)::bigint as store_count,
      count(distinct fs.product_id)::bigint as product_count,
      sum(fs.soldqty)::bigint as soldqty_sum,
      sum(fs.drawqty)::bigint as drawqty_sum,
      avg(case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end) as avg_row_sellthrough
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    where dp.type = 'SIP'
    group by dp.segment, dp.subsegment
    order by soldqty_sum desc
  ",
  holdout_mix = "
    select
      dp.type,
      dp.segment,
      dp.subsegment,
      count(*)::bigint as schedule_rows,
      count(distinct ps.store_id)::bigint as store_count,
      count(distinct ps.product_id)::bigint as product_count
    from holdout.printing_schedule ps
    join holdout.dim_product dp on dp.product_id = ps.product_id
    group by dp.type, dp.segment, dp.subsegment
    order by schedule_rows desc
  ",
  correlation_sample = "
    select
      fs.soldqty::double precision as soldqty,
      fs.drawqty::double precision as drawqty,
      case when fs.drawqty > 0 then fs.soldqty::double precision / fs.drawqty else null end as sellthrough,
      dp.price::double precision as price,
      (dp.offsaledate - dp.onsaledate)::double precision as issue_days,
      ds.facings::double precision as facings,
      ds.pockets::double precision as pockets,
      case when ds.merchandised then 1.0 else 0.0 end as merchandised,
      dg.population::double precision as population,
      dg.age_19_and_under::double precision as age_19_and_under,
      dg.age_30_to_44::double precision as age_30_to_44,
      dg.age_60_and_over::double precision as age_60_and_over,
      dg.less_than_10k::double precision as less_than_10k,
      dg.between_50k_and_74k::double precision as between_50k_and_74k,
      dg.income_200k_or_more::double precision as income_200k_or_more,
      dg.bachelors_degree::double precision as bachelors_degree,
      dg.graduate_or_professional_degree::double precision as graduate_or_professional_degree,
      dg.family_household::double precision as family_household
    from core.fact_sale fs
    join core.dim_product dp on dp.product_id = fs.product_id
    join core.dim_store ds on ds.store_id = fs.store_id
    left join core.demographic dg on dg.postal_code = ds.postal_code
    where mod(fs.store_id, 97) = 0
    limit 50000
  "
)

results <- list()
for (name in names(queries)) {
  log_line("Running query: ", name)
  results[[name]] <- DBI::dbGetQuery(con, queries[[name]])
}

log_line("Writing CSV outputs to ", output_dir)
for (name in names(results)) {
  write_csv_safe(results[[name]], file.path(output_dir, paste0(name, ".csv")))
}

fact_quality <- results$fact_quality
type_summary <- results$type_summary
chain_summary <- results$chain_summary
class_summary <- results$class_summary
region_summary <- results$region_summary
sip_segment_summary <- results$sip_segment_summary
weeklies_title_summary <- results$weeklies_title_summary
correlation_sample <- results$correlation_sample

save_barplot(
  values = type_summary$avg_row_sellthrough,
  labels = type_summary$type,
  main = "Average Sell-Through by Product Type",
  ylab = "Average Row Sell-Through",
  path = file.path(output_dir, "plot_sellthrough_by_type.png"),
  las = 1
)

top_chains <- head(chain_summary[order(-chain_summary$soldqty_sum), ], 15)
save_barplot(
  values = top_chains$avg_row_sellthrough,
  labels = top_chains$store_chain,
  main = "Average Sell-Through by Top Chains",
  ylab = "Average Row Sell-Through",
  path = file.path(output_dir, "plot_top_chain_sellthrough.png")
)

top_classes <- head(class_summary[order(-class_summary$soldqty_sum), ], 12)
save_barplot(
  values = top_classes$avg_row_sellthrough,
  labels = top_classes$classoftrade,
  main = "Average Sell-Through by Class of Trade",
  ylab = "Average Row Sell-Through",
  path = file.path(output_dir, "plot_class_sellthrough.png")
)

top_sip_segments <- head(sip_segment_summary[order(-sip_segment_summary$soldqty_sum), ], 15)
save_barplot(
  values = top_sip_segments$avg_row_sellthrough,
  labels = paste(top_sip_segments$segment, top_sip_segments$subsegment, sep = " | "),
  main = "SIP Segment / Subsegment Sell-Through",
  ylab = "Average Row Sell-Through",
  path = file.path(output_dir, "plot_sip_segment_sellthrough.png")
)

top_weekly_titles <- stats::aggregate(
  list(
    soldqty_sum_total = weeklies_title_summary$soldqty_sum,
    drawqty_sum_total = weeklies_title_summary$drawqty_sum
  ),
  by = list(title = weeklies_title_summary$title),
  FUN = sum,
  na.rm = TRUE
)
top_weekly_titles$avg_sellthrough <- ifelse(
  top_weekly_titles$drawqty_sum_total > 0,
  top_weekly_titles$soldqty_sum_total / top_weekly_titles$drawqty_sum_total,
  NA_real_
)
top_weekly_titles <- head(top_weekly_titles[order(-top_weekly_titles$soldqty_sum_total), , drop = FALSE], 12)
save_barplot(
  values = top_weekly_titles$avg_sellthrough,
  labels = top_weekly_titles$title,
  main = "Weeklies Title Sell-Through",
  ylab = "Aggregate Sell-Through",
  path = file.path(output_dir, "plot_weeklies_title_sellthrough.png")
)

png(file.path(output_dir, "plot_soldqty_histogram.png"), width = 1600, height = 900, res = 140)
hist(
  correlation_sample$soldqty,
  breaks = 100,
  main = "Sampled SoldQty Distribution",
  xlab = "SoldQty",
  col = "steelblue",
  border = "white"
)
dev.off()

numeric_sample <- correlation_sample[, vapply(correlation_sample, is.numeric, logical(1)), drop = FALSE]
numeric_sample <- numeric_sample[, colSums(!is.na(numeric_sample)) > 0, drop = FALSE]
correlation_matrix <- stats::cor(numeric_sample, use = "pairwise.complete.obs")
write_csv_safe(
  data.frame(feature = rownames(correlation_matrix), correlation_matrix, row.names = NULL, check.names = FALSE),
  file.path(output_dir, "numeric_correlation_matrix.csv")
)
save_heatmap(correlation_matrix, file.path(output_dir, "plot_numeric_correlation_heatmap.png"), "Numeric Feature Correlation Heatmap")

summary_lines <- c(
  paste("EDA output directory:", output_dir),
  "",
  "Key facts:",
  paste("- core.fact_sale rows:", format(fact_quality$row_count, big.mark = ",")),
  paste("- negative soldqty rows:", format(fact_quality$negative_soldqty_rows, big.mark = ",")),
  paste("- zero soldqty rows:", format(fact_quality$zero_soldqty_rows, big.mark = ",")),
  paste("- stockout proxy rows:", format(fact_quality$stockout_proxy_rows, big.mark = ",")),
  paste("- total soldqty:", format(fact_quality$total_soldqty, big.mark = ",")),
  paste("- total drawqty:", format(fact_quality$total_drawqty, big.mark = ",")),
  "",
  "Top-level outputs:",
  "- table_count.csv",
  "- fact_quality.csv",
  "- type_summary.csv",
  "- chain_summary.csv",
  "- class_summary.csv",
  "- region_summary.csv",
  "- store_mix_summary.csv",
  "- weeklies_title_summary.csv",
  "- sip_segment_summary.csv",
  "- holdout_mix.csv",
  "- numeric_correlation_matrix.csv"
)
writeLines(summary_lines, con = file.path(output_dir, "README.txt"))

log_line("EDA complete. Outputs written to ", output_dir)
