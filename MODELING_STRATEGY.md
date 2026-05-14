# PaperRush Modeling Strategy

## Objective

The real business objective is not only to predict `SoldQty`, but to improve profitability by recommending a better `DrawQty` under asymmetric stockout and oversupply costs.

So the solution should be split into two layers:
- demand forecasting: estimate expected sales for each `store_id x product_id`
- draw optimization: convert the forecast into a profit-aware delivery recommendation

## Prediction Grain

- training grain: one row per historical `store_id x product_id`
- scoring grain: one row per July `store_id x product_id` from the printing schedule
- time must be reconstructed from `DIM_PRODUCT.onsaledate` and `offsaledate`
- `PRODUCT_ID` represents an issue instance, not a timeless title

## Product-Family Modeling

The case should not be modeled with one single logic from day one.

- `Weeklies`
  - stronger recurring title behavior
  - more useful title/store historical continuity
  - should lean on lagged historical sales patterns
- `SIP`
  - issue-level novelty is much stronger
  - segment, subsegment, content similarity, and embeddings matter more
  - needs analog-based similarity features in addition to tabular history

## Modeling Stack

### 1. Baselines

Start with business-sane benchmarks:
- historical average sales by store/title
- historical average by store/segment
- average sell-through rules
- chain/class-of-trade/regional averages

These baselines define the minimum acceptable performance and provide business intuition.

### 2. Family-Specific Models

- `Weeklies`: boosted tabular models over historical and store/title features
- `SIP`: boosted tabular models plus similarity features derived from segment and embedding neighbors

### 3. Ensemble Layer

Blend complementary models only after the individual families are stable:
- heuristic baseline
- boosted tabular model
- similarity model for SIP

Use out-of-fold predictions for blending rather than naive averaging.

## EDA Priorities

Do not waste time on generic broad correlation exercises without business framing. Focus on:

### Store Performance
- total sold, total draw, sell-through
- performance by `store_chain`, `classoftrade`, `region`
- effect of `merchandised`, `facings`, `pockets`
- store product mix richness

### Weeklies Performance
- title-level performance over issue chronology
- store-title persistence
- trend by `onsaledate`
- sensitivity to merchandising and store attributes

### SIP Performance
- segment and subsegment performance
- similarity between issues using embeddings
- usefulness of analog history at store, chain, and category levels

## Feature Strategy

### Store Features
- `store_chain`, `classoftrade`, `region`
- `merchandised`, `facings`, `pockets`
- demographic profile
- store historical aggregates:
  - average sold
  - average draw
  - average sell-through
  - zero-sales rate
  - stockout-like rate

### Product Features
- `type`, `title`, `segment`, `subsegment`
- `price`
- `frequency`
- `onsaledate`, `offsaledate`
- issue duration

### Interaction Features
- store x title history
- store x segment history
- store x subsegment history
- chain x title
- chain x segment
- store x type

These interaction features are likely more important than many raw columns.

## Embedding Strategy

Do not dump raw 384-dimensional vectors into the first-pass model and hope for the best.

First use embeddings to create analog features:
- nearest historical SIP issues in embedding space
- average sales of top-k similar issues
- average sales of top-k similar issues in same segment
- average sales of similar issues in same store or chain if available

Only later consider PCA-reduced embedding features if they add measurable value.

## Target and Censoring

Primary forecast target:
- `SoldQty`

But observed sales may be censored when `SoldQty == DrawQty` and demand exceeds supply.

So EDA and validation should explicitly track:
- stockout-like rows: `drawqty > 0 and soldqty == drawqty`
- oversupply: `drawqty - soldqty`
- negative sales adjustments

## Validation

Use time-based validation only.

- sort by `onsaledate`
- train on earlier issues, validate on later issues
- use rolling or expanding folds
- keep July untouched as the final holdout / pilot scoring set

Random splits would leak future issue behavior backward and are not acceptable here.

## Metrics

Use both predictive and business metrics.

### Predictive
- MAE
- RMSE
- WAPE

### Business
- expected revenue
- stockout loss
- oversupply / waste cost
- pilot profit uplift versus baseline draw logic

## Draw Recommendation

Do not set `DrawQty` equal to the point forecast mechanically.

Use the forecast distribution or quantile forecasts and choose the draw that maximizes expected profit:
- test `p50`, `p75`, `p90`
- compare draw rules under business cost assumptions

## Recommended Execution Order

1. reconstruct issue chronology from `onsaledate`
2. run business-focused EDA
3. build baseline heuristics
4. engineer hierarchical fallback features
5. train `Weeklies` demand model
6. train `SIP` demand + similarity model
7. build restrained ensemble
8. optimize draw recommendation
9. score July printing schedule
10. evaluate pilot once actual sales are available

## Current Data Caveats

- `DEMOGRAPHICS` is provisionally deduplicated by `postal_code` using `MAX(population)` for modeling
- historical fact data contains negative `SoldQty` rows, which likely reflect adjustments or returns
- July remains a pure holdout / pilot universe and must not enter model training
