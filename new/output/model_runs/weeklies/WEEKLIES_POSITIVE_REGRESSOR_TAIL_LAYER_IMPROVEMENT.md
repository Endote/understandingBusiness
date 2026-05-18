# Weeklies Full-Label July Holdout Pipeline

This document is the current Weeklies production-benchmark record. It describes the actual full-label training and July holdout scoring pipeline used in the latest run, grounded in the current codebase and generated run artifacts.

## Final Artifacts

- Full-label amount and tail models: `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412`
- Full-label pruned incidence classifier: `new/output/JulyScoring/weeklies_full_deployment_models/incidence_pruned/20260518_005200`
- July holdout scoring run: `new/output/JulyScoringFullLabel/weeklies/20260518_010259`
- Final two-tier prediction file: `new/output/JulyScoringFullLabel/weeklies/20260518_010259/twotier_gate_grid/july_holdout_weeklies_predictions_twotier_best.parquet`
- Final ceiled comparison report: `new/output/JulyScoringFullLabel/weeklies/20260518_010259/final_comparison_report`
- Final group-alignment report: `new/output/JulyScoringFullLabel/weeklies/20260518_010259/ceiled_group_alignment`

## Pipeline Shape

The final scoring pipeline is a two-stage sales system:

1. Train a positive-only amount regressor on all labeled Weeklies rows where sales are positive.
2. Train a positive-only tail layer to reshape the amount regressor toward high-positive and low-positive behavior.
3. Train an all-row incidence classifier on all labeled Weeklies rows to estimate whether a store/product row should sell at all.
4. Score July holdout rows with the amount stack and incidence classifier.
5. Apply a two-tier raw-probability incidence gate and amount scale.
6. Apply `ceil` to final predictions for integer-unit benchmark reporting.

The positive regressor is intentionally not trained on zero/negative rows. The incidence classifier handles row inclusion. This avoids forcing the regressor to learn both sale incidence and positive amount magnitude.

## Data Scope

The full-label training data comes from the current modeled Weeklies datasets under:

- `new/output/modeling_datasets/weeklies/regressor_positive.parquet`
- `new/output/modeling_datasets/weeklies/classifier_all.parquet`

Training row counts from current manifests and summaries:

- Positive amount rows: `2,060,572`
- All-row incidence rows: `5,688,376`
- Historical all-row positive rate: `0.3622425795`
- July holdout rows scored: `312,094`
- July holdout products: `115`
- July holdout stores: `26,960`
- July holdout onsale range: `2024-07-01` to `2024-07-22`

## Base Row Features

The holdout scoring frame is built from `holdout.printing_schedule`, `holdout.dim_product`, `holdout.dim_store`, and `holdout.demographic`.

Base product/store/calendar/display fields:

- Keys: `store_id`, `product_id`, `onsaledate`
- Product dates and identity: `offsaledate`, `title`, `type`, `segment`, `subsegment`, `frequency`
- Store context: `store_chain`, `region`, `classoftrade`
- Product economics/calendar: `price`, `issue_length_days`, `onsale_month`, `onsale_month_cat`, `onsale_week`, `onsale_dow`
- Display/store execution: `merchandised`, `facings`, `pockets`
- Demographic columns from `build_modeling_datasets.BASE_DEMOGRAPHIC_COLUMNS`

Categorical model features for both amount and incidence contracts:

- `title`
- `store_id`
- `onsale_month_cat`
- `segment`
- `subsegment`
- `frequency`
- `store_chain`
- `region`
- `classoftrade`

## Derived Feature Families

The full feature contract before model-specific pruning has `552` numeric features and `9` categorical features.

Numeric feature families used by the amount contract:

- Base row numeric fields: `8`
- Completed-history priors: `188`
- Log transforms: `4`
- Ratio/capacity features: `29`
- Embedding PCA features: `130`
- Embedding analog features: `24`
- Embedding affinity features: `182`
- Demographic-like numeric fields: `7`
- Other numeric derived fields: `35`

Completed priors are leakage-controlled. Historical products must have `offsaledate` strictly before the current row `onsaledate`. The prior families include:

- Global
- Store
- Store/title
- Store/segment
- Store/subsegment
- Store/type
- Chain
- Class-of-trade
- Chain/segment
- Chain/class/subsegment
- Chain/title
- Class/subsegment
- Class/title
- Title
- Segment
- Subsegment
- Same-month variants
- Trailing completed windows of `90`, `180`, and `365` days
- Recency-weighted completed priors with half-life `180` days

Embedding feature families:

- PCA of 384-dimensional product embeddings into `130` components.
- Retrieval analog features from top-`10` prior completed products by cosine similarity.
- Shrunk analog sales statistics using same-subsegment prior aggregates.
- Tail-vs-low centroid affinity features by context.

Affinity contexts:

- Global
- Store
- Store/segment
- Store/subsegment
- Store/title
- Chain/title
- Class/title
- Chain/segment
- Chain/class/subsegment
- Class/subsegment
- Chain
- Class
- Segment
- Subsegment

For the July holdout run, every holdout product had an embedding:

- Amount-stage holdout rows with embedding: `312,094 / 312,094`
- Incidence-stage holdout rows with embedding: `312,094 / 312,094`
- Holdout products with prior analog neighbors: `115 / 115`

## Amount Model

Script:

```bash
python3 new/train_full_weeklies_deployment.py
```

Final amount artifact:

- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/regressor_positive`

Training scope:

- All labeled positive Weeklies rows only.
- Rows: `2,060,572`

Reference model copied for settings:

- `new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_205934`

Objective:

- `asym_curve_log1p`

Weighting:

- `category_xtrade_balance`

XGBoost parameters:

- `max_depth=7`
- `min_child_weight=10`
- `eta=0.04`
- `subsample=0.85`
- `colsample_bytree=0.85`

Asymmetric objective decisions:

- Bottom deciles penalize overprediction heavily.
- D7-D8 penalize underprediction very heavily.
- D9-D10 still penalize underprediction, but also penalize overprediction enough to avoid runaway top-tail inflation.
- Total-ratio penalty discourages q90-style global unit inflation.
- Stockout proxy increases underprediction pressure where stockout-like behavior suggests censored demand.

Objective configuration:

- `asym_bottom_over_weight=12.0`
- `asym_bottom_under_weight=0.1`
- `asym_top_under_weight=4.0`
- `asym_top_over_weight=0.4`
- `asym_d7_d8_under_weight=14.0`
- `asym_d7_d8_over_weight=0.3`
- `asym_d9_d10_under_weight=3.0`
- `asym_d9_d10_over_weight=1.5`
- `asym_total_ratio_penalty=0.75`
- `stockout_under_weight=3.0`
- `stockout_over_weight=0.5`

Model-specific feature handling:

- The amount model uses the full Weeklies amount feature contract after standard drop rules.
- Dropped active numeric feature: `onsale_week`
- Numeric features after drops: `551`
- Categorical features after drops: `9`

## Tail Layer

Script:

```bash
python3 new/train_full_weeklies_deployment.py
```

Final tail artifact:

- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/tail_layer`

Training scope:

- Same all-labeled positive Weeklies rows as the amount model.

Reference run:

- `new/output/tail_layer_runs/weeklies/regressor_positive/tail_q70/20260517_210107`

Tail target:

- `tail_q70`
- `tail_quantile=0.70`

Low layer:

- Enabled.
- Low target: `sales_target <= 1.0`

Selected multiplier:

- `alpha=2.0`
- `band_alpha=0.0`
- `external_alpha=0.0`
- `low_alpha=1.2`
- `scale=0.45`

Interpretation:

- The base regressor predicts positive amount.
- The tail classifier estimates whether the row is in the upper positive-sale region.
- The low classifier estimates whether the row belongs to the low-positive region.
- The multiplier reshapes the positive amount prediction before incidence gating.

## Incidence Classifier

Script:

```bash
python3 new/train_full_weeklies_incidence_pruned.py
```

Final incidence artifact:

- `new/output/JulyScoring/weeklies_full_deployment_models/incidence_pruned/20260518_005200`

Training scope:

- All labeled Weeklies rows.
- Rows: `5,688,376`
- Target: `positive_sale_flag`
- Positive rate: `0.3622425795`

Reference run:

- `new/output/model_runs/weeklies/classifier_all/binary_logistic/20260517_211136`

Objective:

- `binary:logistic`

XGBoost parameters:

- `max_depth=6`
- `min_child_weight=50`
- `eta=0.05`
- `subsample=0.85`
- `colsample_bytree=0.85`
- `scale_pos_weight=1.7605810425`
- Metrics: `logloss`, `aucpr`, `auc`

Feature decision:

- The final incidence classifier is pruned for generalization and runtime.
- It uses the top `160` numeric features by reference gain, plus important fallback numeric fields and all categorical features.
- Final incidence feature count: `162` numeric, `9` categorical, `28,849` encoded feature names.

Incidence feature families retained by the pruned model:

- Base row numeric fields: `7`
- Completed-history priors: `118`
- Log transforms: `2`
- Ratio/capacity features: `11`
- Embedding analog features: `23`
- Other selected numeric fields: `11`
- All `9` categorical fields

The pruned incidence model does not use raw PCA or affinity features in the final selected artifact. That is deliberate: the full feature universe was available, but the deployed incidence classifier keeps the strongest gain-ranked completed-prior, analog, base, and categorical signals.

## July Holdout Scoring

Script:

```bash
python3 new/score_july_holdout_weeklies.py --output-dir new/output/JulyScoringFullLabel --amount-run-dir new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/regressor_positive --tail-run-dir new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/tail_layer --classifier-run-dir new/output/JulyScoring/weeklies_full_deployment_models/incidence_pruned/20260518_005200 --calibration-mode none --pipeline-variant G_soft_gate_amount_raw_t0.65_floor0.20_low0.85_mid1.10_top0.95_scale0.80
```

Initial scoring variant from `run_summary.json`:

- `G_soft_gate_amount_raw_t0.65_floor0.20_low0.85_mid1.10_top0.95_scale0.80`

Initial scoring formula:

```text
positive_prediction = tail_adjusted_positive_regressor_prediction
incidence_source = raw incidence probability
gate = incidence_source >= 0.65
incidence_layer = 0.20 + 0.80 * gate
amount_multiplier = rank_bucket_multiplier(positive_prediction, low=0.85, mid=1.10, top=0.95, global_scale=0.80)
final_prediction = positive_prediction * incidence_layer * amount_multiplier
```

Calibration mode was `none` for this final run. The pipeline used raw incidence probability for the selected full-label holdout benchmark.

## Two-Tier Gate Selection

Script:

```bash
python3 new/optimize_july_twotier_gate.py --scoring-dir new/output/JulyScoringFullLabel/weeklies/20260518_010259
```

Selected variant:

- `twotier_raw_low0.475_high0.625_floor0.25_scale1.20`

Selected two-tier configuration:

- Raw incidence probability source.
- Low threshold: `0.475`
- High threshold: `0.625`
- Floor: `0.25`
- Global scale: `1.20`

Selected pre-ceil summary:

- Rows: `312,094`
- Nonzero rate: `0.375345`
- Unit sum: `433,502.738`
- Mean all rows: `1.3890`
- Mean nonzero rows: `3.7006`
- P90: `4.0868`
- P95: `8.2596`
- P99: `15.4218`
- Top-10 share: `0.6756`

The two-tier gate was chosen because the earlier hard gate underfed D7-D8 and concentrated positive predictions too harshly into the top rows. The two-tier version keeps a suppression floor while allowing moderate-probability rows to carry units.

## Final Integer Benchmark

Final report command:

```bash
python3 new/report_ceiled_prediction_deciles.py --old-scoring-dir new/output/JulyScoring/weeklies/20260517_235821 --new-scoring-dir new/output/JulyScoringFullLabel/weeklies/20260518_010259
```

Final human-facing benchmark applies:

```text
ceiled_prediction = ceil(july_holdout_prediction_twotier_best)
```

Final ceiled summary:

| Series | Rows | Nonzero Rate | Unit Sum | Mean All | Mean Nonzero | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old July hard gate ceil | 312,094 | 0.1851 | 461,657 | 1.4792 | 7.9909 | 5 | 10 | 19 |
| Full-label two-tier soft gate ceil | 312,094 | 0.3753 | 495,011 | 1.5861 | 4.2257 | 5 | 9 | 16 |

Final decile means:

| Decile | Full Labeled Actual | Model Train Split Actual | Old July Hard Gate Ceil | Full-Label Two-Tier Soft Gate Ceil |
|---|---:|---:|---:|---:|
| D1 | 0.000 | 0.000 | 0.000 | 0.000 |
| D2 | 0.000 | 0.000 | 0.000 | 0.000 |
| D3 | 0.000 | 0.000 | 0.000 | 0.000 |
| D4 | 0.000 | 0.000 | 0.000 | 0.000 |
| D5 | 0.000 | 0.000 | 0.000 | 0.000 |
| D6 | 0.000 | 0.000 | 0.000 | 0.000 |
| D7 | 0.622 | 0.664 | 0.000 | 0.753 |
| D8 | 1.346 | 1.407 | 0.000 | 1.511 |
| D9 | 2.786 | 2.910 | 3.282 | 3.701 |
| D10 | 9.875 | 10.716 | 11.510 | 9.895 |

Final decile medians:

| Decile | Full Labeled Actual | Model Train Split Actual | Old July Hard Gate Ceil | Full-Label Two-Tier Soft Gate Ceil |
|---|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 |
| D2 | 0 | 0 | 0 | 0 |
| D3 | 0 | 0 | 0 | 0 |
| D4 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 0 | 0 |
| D6 | 0 | 0 | 0 | 0 |
| D7 | 1 | 1 | 0 | 1 |
| D8 | 1 | 1 | 0 | 1 |
| D9 | 3 | 3 | 4 | 4 |
| D10 | 7 | 8 | 10 | 9 |

## Distribution Alignment Diagnostics

Command:

```bash
python3 new/analyze_ceiled_group_alignment.py --scoring-dir new/output/JulyScoringFullLabel/weeklies/20260518_010259
```

The final ceiled prediction aligns well at broad segment/subsegment level, but it is not perfectly aligned by title. The main drift is predicted unit intensity, not row mix.

| Grouping | Unit Share TVD | Max Unit Shift | Biggest Shift |
|---|---:|---:|---|
| segment | 0.0572 | 5.7205 pp | WOMENS |
| subsegment | 0.0572 | 5.7205 pp | EVERYDAY LIVING |
| classoftrade | 0.0848 | 8.4634 pp | SM CHAIN |
| store_chain | 0.0960 | 4.0841 pp | Chain 9 |
| title | 0.1394 | 13.9383 pp | Simply Woman |
| title_classoftrade | 0.1660 | 10.9005 pp | Simply Woman \| SM CHAIN |

Important known bias:

- `Simply Woman` is overallocated versus full labeled actual unit share.
- `Everyday Bloom` is underallocated.
- `Simply Woman | SM CHAIN` is the largest title/channel over-shift.

This is the main remaining explainability note. The pipeline gets the all-row decile shape much closer than the previous hard gate, but title/channel intensity calibration is the next natural improvement area.

## Required Reproduction Files

Core scripts:

- `new/build_modeling_datasets.py`
- `new/build_incidence_dataset.py`
- `new/train_stage_model.py`
- `new/train_tail_layer.py`
- `new/train_incidence_classifier.py`
- `new/calibrate_incidence_classifier.py`
- `new/evaluate_incidence_pipeline.py`
- `new/train_full_weeklies_deployment.py`
- `new/train_full_weeklies_incidence_pruned.py`
- `new/score_july_holdout_weeklies.py`
- `new/optimize_july_twotier_gate.py`
- `new/report_ceiled_prediction_deciles.py`
- `new/analyze_ceiled_group_alignment.py`

Dataset and feature artifacts:

- `new/output/modeling_datasets/weeklies/regressor_positive.parquet`
- `new/output/modeling_datasets/weeklies/classifier_all.parquet`
- `new/output/modeling_datasets/weeklies/manifest_regressor_positive.json`
- `new/output/modeling_datasets/weeklies/manifest_classifier_all.json`
- `new/output/modeling_datasets/weeklies/feature_contract_regressor_positive.json`
- `new/output/modeling_datasets/weeklies/feature_contract_classifier_all.json`
- `new/output/modeling_datasets/weeklies/embedding_pca_regressor_positive.json`
- `new/output/modeling_datasets/weeklies/embedding_pca_classifier_all.json`

Reference model artifacts used to transfer settings:

- `new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_205934`
- `new/output/tail_layer_runs/weeklies/regressor_positive/tail_q70/20260517_210107`
- `new/output/model_runs/weeklies/classifier_all/binary_logistic/20260517_211136`

Final full-label model artifacts:

- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/regressor_positive/model.json`
- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/regressor_positive/encoding_artifact.json`
- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/tail_layer/tail_classifier.json`
- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/tail_layer/low_classifier.json`
- `new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/tail_layer/tail_encoding_artifact.json`
- `new/output/JulyScoring/weeklies_full_deployment_models/incidence_pruned/20260518_005200/model.json`
- `new/output/JulyScoring/weeklies_full_deployment_models/incidence_pruned/20260518_005200/encoding_artifact.json`

Database inputs required at scoring time:

- `holdout.printing_schedule`
- `holdout.dim_product`
- `holdout.dim_store`
- `holdout.demographic`
- `holdout.content_embedding`
- Current labeled core tables used by the feature builders, including product, store, sales, demographics, and content embedding tables referenced by `build_modeling_datasets.py`

## Reproduction Commands

Build/refresh modeling datasets:

```bash
python3 new/build_modeling_datasets.py --family Weeklies
```

Build/refresh all-row incidence dataset:

```bash
python3 new/build_incidence_dataset.py --family Weeklies
```

Train full-label amount and tail artifacts:

```bash
python3 new/train_full_weeklies_deployment.py
```

Train final pruned full-label incidence classifier:

```bash
python3 new/train_full_weeklies_incidence_pruned.py
```

Score July holdout with the full-label artifacts:

```bash
python3 new/score_july_holdout_weeklies.py --output-dir new/output/JulyScoringFullLabel --amount-run-dir new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/regressor_positive --tail-run-dir new/output/JulyScoring/weeklies_full_deployment_models/20260518_003412/models/tail_layer --classifier-run-dir new/output/JulyScoring/weeklies_full_deployment_models/incidence_pruned/20260518_005200 --calibration-mode none --pipeline-variant G_soft_gate_amount_raw_t0.65_floor0.20_low0.85_mid1.10_top0.95_scale0.80
```

Optimize the final two-tier gate over the scored holdout distribution:

```bash
python3 new/optimize_july_twotier_gate.py --scoring-dir new/output/JulyScoringFullLabel/weeklies/20260518_010259
```

Generate ceiled decile comparisons:

```bash
python3 new/report_ceiled_prediction_deciles.py --old-scoring-dir new/output/JulyScoring/weeklies/20260517_235821 --new-scoring-dir new/output/JulyScoringFullLabel/weeklies/20260518_010259
```

Generate group-alignment diagnostics:

```bash
python3 new/analyze_ceiled_group_alignment.py --scoring-dir new/output/JulyScoringFullLabel/weeklies/20260518_010259
```
