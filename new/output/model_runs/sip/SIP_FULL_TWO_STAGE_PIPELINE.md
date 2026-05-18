# SIP Full Holdout Prediction Pipeline

Generated: 2026-05-18

This document records the current SIP pipeline that produced the best full July holdout predictions. It is grounded in the current code, database-derived run artifacts, and current output files.

## Final Recommendation

Use the full-label SIP deployment model run plus the integer-aware postprocessed candidate `ceilcap_9479`.

Final prediction artifact:

`new/output/JulyScoringFullLabel/sip/20260518_100055/soft_gate_distribution_grid/20260518_131023/best/july_holdout_sip_predictions.parquet`

Final prediction column:

`july_holdout_prediction`

For integer sold-unit delivery, use:

`ceil(july_holdout_prediction)`

## Final Artifacts

| artifact | path |
|---|---|
| full-label model run | `new/output/JulyScoringFullLabel/sip_full_deployment_models/20260518_080836` |
| base July scoring run | `new/output/JulyScoringFullLabel/sip/20260518_100055` |
| best postprocessor run | `new/output/JulyScoringFullLabel/sip/20260518_100055/soft_gate_distribution_grid/20260518_131023` |
| best full holdout predictions | `new/output/JulyScoringFullLabel/sip/20260518_100055/soft_gate_distribution_grid/20260518_131023/best/july_holdout_sip_predictions.parquet` |
| ceiled alignment report | `new/output/JulyScoringFullLabel/sip/20260518_100055/soft_gate_distribution_grid/20260518_131023/best/latest_prediction_group_alignment_ceil` |

## Dataset Scope

Full labeled SIP classifier dataset:

| metric | value |
|---|---:|
| rows | 10,870,298 |
| positive rate | 0.230501 |
| train rows | 7,936,804 |
| validation rows | 1,294,369 |
| test rows | 1,639,125 |
| target mean | 0.460006 |
| target median | 0 |
| target p90 | 2 |
| target p99 | 5 |

Positive-only SIP regressor dataset:

| metric | value |
|---|---:|
| rows | 2,505,611 |
| target mean | 1.995680 |
| target median | 2 |
| target p90 | 4 |
| target p99 | 7 |

July SIP holdout:

| metric | value |
|---|---:|
| rows | 449,437 |
| products | 183 |
| stores | 25,180 |
| onsale date min | 2024-06-28 |
| onsale date max | 2024-07-22 |

## Feature Families

The amount and incidence stages use the same high-level feature contract:

| feature type | count / fields |
|---|---|
| categorical | `store_id`, `onsale_month_cat`, `segment`, `subsegment`, `frequency`, `store_chain`, `region`, `classoftrade` |
| numeric | 552 numeric features |
| scoring-unavailable exclusions | raw future/realized scoring fields such as final sales/draw/completion fields are excluded |

Core feature families:

- completed-history priors with leakage fixed by requiring historical product `offsaledate` to be strictly before the scored product `onsaledate`;
- global, store, store-title, store-segment, store-subsegment, chain, class, segment, subsegment, and chain/class/subsegment priors;
- trailing 90/180/365 day priors;
- recency-weighted priors with 180 day half-life;
- same-month seasonal priors;
- product embedding features and analog/affinity features rebuilt for amount and incidence stages;
- derived numeric transforms added at scoring time;
- `product_id` is not used as a memorization feature; `store_id` is retained.

## Model Stack

Full-label deployment training produced these models:

| component | rows | rounds | objective |
|---|---:|---:|---|
| positive amount base | 2,505,611 | 79 | `quantile_40` |
| positive Tweedie signal | 2,505,611 | 293 | `tweedie` |
| tail layer | 2,505,611 | 180 / 180 / 180 | broad tail, band, low classifiers |
| base incidence | 10,870,298 | 64 | `binary:logistic` |
| avg-sales incidence | 10,870,298 | 153 | `binary:logistic` |
| amount-band classifier | 10,870,298 | 36 | five-class `0,1,2,3-4,5+` |
| cumulative incidence | 10,870,298 | 87 / 89 / 187 / 87 / 218 | `P(sales>=1..5)` |
| zero-leak veto | 1,029,094 | 178 | binary veto |
| low-positive veto | 1,029,094 | 192 | binary `sales<=1` veto |

The positive amount stack starts from q40, then applies the SIP tail multiplier:

| parameter | value |
|---|---:|
| `alpha` | 5.4 |
| `band_alpha` | 0.0 |
| `external_alpha` | 0.4 |
| `low_alpha` | 0.6 |
| `scale` | 0.345 |

The base full scoring run writes `sip_prediction_892` as the default `july_holdout_prediction`. The final accepted result replaces that with the integer-aware soft distribution candidate `ceilcap_9479`.

## Best Postprocessor

Best candidate:

`candidate_id=9479`, `variant_family=soft_rank`

Important parameters:

| parameter | value |
|---|---:|
| calibration group | `segment_classoftrade` |
| calibration strength | 0.65 |
| budget strategy | `segment_subsegment` |
| budget strength | 0.408985 |
| budget iterations | 3 |
| low quantile | 0.683553 |
| high quantile | 0.889116 |
| mid floor | 0.199445 |
| low floor | 0.0 |
| global scale | 0.513321 |
| D8/D9 boost | 0.175035 |
| top damp | 0.397725 |
| raw PUZZLES cap | 0.42 |
| raw SUDOKU cap | 0.18 |
| raw WORD SEEK cap | 0.26 |
| ceiled PUZZLES cap | 0.32 |
| ceiled SUDOKU cap | 0.11 |
| ceiled WORD SEEK cap | 0.14 |

The key final improvement was optimizing after integer ceiling. Raw float predictions can look acceptable while becoming structurally wrong after `ceil()`. The final allocator directly controls ceiled nonzero rate, D8/D9/D10 shape, and large overrepresented groups.

## Reproduction Commands

Train full-label SIP deployment models:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/train_full_sip_deployment.py
```

Score full July SIP holdout:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/score_july_holdout_sip.py --model-run-dir new/output/JulyScoringFullLabel/sip_full_deployment_models/20260518_080836
```

Run the final integer-aware postprocessor grid:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/optimize_sip_soft_gate_distribution.py --scoring-dir new/output/JulyScoringFullLabel/sip/20260518_100055 --trials 10000 --screen-sample-rows 150000 --screen-top-k 1000 --top-k 30 --objective-profile integer_ratio_groups
```

Regenerate the final ceiled distribution report:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/analyze_sip_latest_group_alignment.py --scoring-dir new/output/JulyScoringFullLabel/sip/20260518_100055/soft_gate_distribution_grid/20260518_131023/best --prediction-column july_holdout_prediction --ceil-prediction
```

## Final Holdout Summary

Comparison is full labeled SIP actuals versus ceiled July holdout predictions from `ceilcap_9479`.

| series | rows | nonzero rate | units | mean all | mean nonzero | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full labeled actual | 10,870,298 | 0.230501 | 5,000,399 | 0.460006 | 1.995680 | 2 | 3 | 5 |
| July prediction ceiled | 449,437 | 0.230562 | 240,970 | 0.536160 | 2.325449 | 2 | 2 | 7 |

The nonzero rate is effectively matched. The ceiled prediction is still higher on total mean and positive-row mean, mainly because D10 is heavy.

## Decile Comparison

Unit-share PSI by decile: `0.0191`.

| decile | actual units | pred units | actual share | pred share | delta pp | actual mean | pred mean | actual median | pred median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D2 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D3 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D4 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D5 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D6 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D7 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0 | 0 |
| D8 | 331,551 | 13,735 | 0.0663 | 0.0570 | -0.93 | 0.3050 | 0.3056 | 0 | 0 |
| D9 | 1,268,160 | 48,426 | 0.2536 | 0.2010 | -5.26 | 1.1666 | 1.0775 | 1 | 1 |
| D10 | 3,400,688 | 178,809 | 0.6801 | 0.7420 | +6.20 | 3.1284 | 3.9785 | 3 | 2 |

Interpretation:

- D8 is essentially exact.
- D9 is slightly underfed.
- D10 is overconcentrated, but within an acceptable current frontier.
- Lower seven deciles remain zero, matching the sparse SIP target structure.

## Population Stability View

Group-share PSI is high because the July holdout categorical mix differs from the full historical labeled population. That means these are diagnostics, not hard rejection criteria.

| grouping | groups | unit TVD | unit PSI | row PSI | worst shift |
|---|---:|---:|---:|---:|---|
| segment | 17 | 0.2132 | 2.0416 | 2.9155 | PUZZLES |
| subsegment | 53 | 0.4398 | 4.6461 | 5.7222 | CELEBRITY PROFILE |
| classoftrade | 16 | 0.2536 | 0.6079 | 0.1106 | DRUG CHAIN |
| store_chain | 105 | 0.3491 | 1.3525 | 0.2071 | Chain 4 |
| segment_classoftrade | 244 | 0.3835 | 3.0298 | 2.7354 | PUZZLES / DSCT CHAIN |

Largest remaining unit-share shifts:

| group | actual share | predicted share | delta pp |
|---|---:|---:|---:|
| PUZZLES | 0.2141 | 0.3200 | +10.59 |
| HUMAN CULTURE | 0.0836 | 0.0000 | -8.36 |
| HOME | 0.0699 | 0.0000 | -6.99 |
| CELEBRITY PROFILE | 0.0575 | 0.1682 | +11.07 |
| MUSIC | 0.0893 | 0.0000 | -8.93 |
| DRUG CHAIN | 0.2436 | 0.0317 | -21.19 |
| SM CHAIN | 0.4931 | 0.6559 | +16.28 |
| DSCT CHAIN | 0.2103 | 0.2999 | +8.96 |

Important caveat: some underrepresented historical groups have zero July holdout rows. They cannot be fixed by prediction weighting because they are absent from the scoring population.

## Current Status

The full July SIP holdout is scored and saved. Nothing else is required to have full SIP holdout predictions.

The only optional delivery step is a slim export with identifiers plus final prediction and ceiled integer units.

Suggested final export fields:

- `store_id`
- `product_id`
- `onsaledate`
- `title`
- `segment`
- `subsegment`
- `store_chain`
- `classoftrade`
- `july_holdout_prediction`
- `july_holdout_prediction_ceiled`

## Remaining Known Limitations

- Direct comparison to full historical categorical shares is not apples-to-apples because July holdout mix differs.
- D10 is still heavier than historical actuals.
- Group PSI remains high, especially subsegment and segment-classoftrade.
- Further improvement should focus on non-categorical rank calibration or a July-available expected distribution, not blunt group weighting.
