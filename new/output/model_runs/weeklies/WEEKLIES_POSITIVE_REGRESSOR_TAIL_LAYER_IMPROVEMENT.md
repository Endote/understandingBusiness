# Weeklies Positive-Only Regressor Tail-Layer Improvement

Created: 2026-05-17 12:39:57 CEST

This note records the grounded Weeklies positive-only regressor improvement achieved in the 2026-05-17 experiment loop. It is based on the current codebase and generated model artifacts, not persisted planning notes.

## Manual Run

```
python3 new/build_modeling_datasets.py --family Weeklies
```

```
python3 new/train_stage_model.py --family Weeklies --objective asym_curve_log1p --weighting category_xtrade_balance --rounds 500 --early-stopping 45 --skip-folds --max-depth 6 --min-child-weight 25 --eta 0.05 --curve-penalty 0.08 --asym-bottom-over-weight 3.5 --asym-bottom-under-weight 0.35 --asym-top-under-weight 3.5 --asym-top-over-weight 0.6 --asym-under-start-decile 6 --asym-total-ratio-penalty 0.0 --stockout-under-weight 1.0 --stockout-over-weight 1.0
```

```
python3 new/train_tail_layer.py --family Weeklies --base-run-dir new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_000228 --tail-quantile 0.70 --rounds 350 --early-stopping 30 --multiplier-mode rank --calibration-group-cols '' --alpha-grid 1.8,2.0,2.2,2.4,2.6,2.8,3.0 --scale-grid 0.45,0.475,0.50,0.525,0.55,0.575,0.60
```


## Objective

The model is trained only on positive-sale rows for Weeklies. Zero/negative/other-sale incidence is intentionally deferred to a later classifier. The current target is a positive-row sales regressor optimized for decile shape:

- D1-D4: overprediction is expensive.
- D7-D10: underprediction is expensive.
- D5-D6: calibration/stability zone.
- Total predicted/actual unit ratio should stay near 1.0.

The run-ranking metric used for selection was:

```text
score =
  2.0 * mean(max(bottom_D1_D4_ratio - 1, 0))
+ 2.5 * mean(max(1 - top_D7_D10_ratio, 0))
+ 0.8 * abs(total_ratio - 1)
+ 0.5 * WAPE
```

## Best Current Run

Best current production candidate by the composite decile score:

```text
new/output/tail_layer_runs/weeklies/regressor_positive/tail_q70/20260517_005103
```

This run uses:

- Base regressor: `new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_000228`
- Tail classifier target: top 30 percent of positive rows (`tail_quantile=0.70`)
- Multiplier mode: rank-based tail probability
- Calibration: global only, no group calibration
- Selected multiplier: `alpha=2.4`, `scale=0.50`

No `title,classoftrade` group calibration is used in the best run. Group-aware calibration improved validation but overfit and hurt test-period shape.

## Test Decile Results

Predicted/actual unit ratio by actual sales decile:

| Decile | Previous Base `20260517_000228` | Best Tail Run `20260517_005103` |
|---:|---:|---:|
| D1 | 2.219 | 1.230 |
| D2 | 2.137 | 1.110 |
| D3 | 2.159 | 1.134 |
| D4 | 1.781 | 0.953 |
| D5 | 1.300 | 0.913 |
| D6 | 1.149 | 0.798 |
| D7 | 0.953 | 0.743 |
| D8 | 0.924 | 0.847 |
| D9 | 0.955 | 1.118 |
| D10 | 0.837 | 1.207 |
| Total | 1.078 | 1.053 |

Metric comparison:

| Metric | Previous Base | Best Tail Run |
|---|---:|---:|
| Composite decile score | 2.6626 | 0.8787 |
| WAPE | 0.4901 | 0.6873 |
| Total predicted/actual units | 1.0779 | 1.0528 |

The improvement is not WAPE-driven. WAPE worsens because the selection metric deliberately prioritizes decile shape and business asymmetry over average absolute unit error.

## What Improved

The main win is bottom-decile suppression without starving the extreme top:

- D1 reduced from `2.219x` to `1.230x`.
- D2 reduced from `2.137x` to `1.110x`.
- D3 reduced from `2.159x` to `1.134x`.
- D4 reduced from `1.781x` to `0.953x`.
- D10 increased from `0.837x` to `1.207x`.
- Total unit ratio moved from `1.0779` to `1.0528`.
- Composite decile score improved from `2.6626` to `0.8787`.

The remaining issue is shape within the upper tail:

- D7 remains underpredicted at `0.743x`.
- D8 remains underpredicted at `0.847x`.
- D9 and D10 are overpredicted at `1.118x` and `1.207x`.

This means the tail signal is useful, but still too sharp: it moves units from the bottom into the highest tail more successfully than into the D7-D8 band.

## Dataset Changes

The Weeklies positive-only modeling dataset was rebuilt after adding completed-only prior and embedding-affinity features.

Dataset after rebuild:

- Rows: `2,060,572`
- Numeric features in contract: `552`
- Affinity / affinity-derived numeric features: `182`

Output artifacts:

```text
new/output/modeling_datasets/weeklies/regressor_positive.parquet
new/output/modeling_datasets/weeklies/feature_contract_regressor_positive.json
new/output/modeling_datasets/weeklies/manifest_regressor_positive.json
new/output/modeling_datasets/weeklies/embedding_pca_regressor_positive.json
```

## Completed-Only Prior Feature Families

The dataset builder now includes completed-history priors with strict leakage control:

```text
historical product offsaledate must be strictly before current row onsaledate
```

Base completed-prior contexts:

- `global`
- `chain`
- `class`
- `store`
- `store_title`
- `store_segment`
- `store_subsegment`
- `store_type`
- `chain_segment`
- `chain_class_subsegment`
- `chain_title`
- `class_subsegment`
- `class_title`
- `title`
- `segment`
- `subsegment`

Windowed completed-prior contexts:

- Windows: `90`, `180`, `365` days
- Contexts: `global`, `store`, `store_segment`, `store_subsegment`, `chain_segment`, `chain_class_subsegment`, `class_subsegment`

Recency-weighted completed-prior contexts:

- Half-life: `180` days
- Contexts: same as windowed priors

Month / season prior contexts:

- `global_month`
- `segment_month`
- `subsegment_month`
- `store_subsegment_month`
- `chain_segment_month`

Each prior family emits positive rate, positive average sales, and observation count features.

## Embedding Analog Features

The pre-existing product embedding analog family remains active:

- Product-level nearest-neighbor analogs within the same family.
- Neighbor count and effective neighbor count.
- Similarity-weighted positive sales statistics.
- Similarity-weighted positive sale rate.
- Weighted p50, p75, p90 positive sales.
- Shrunk analog statistics using same-subsegment and family priors.

These analog features are useful, but by themselves they did not solve the store/channel tail problem. They became more useful when crossed with context affinity and completed priors.

## Embedding Affinity Feature Families

The new affinity system converts product embeddings into store/channel/context tail evidence.

Affinity construction:

- PCA product embeddings are fit on training-split products only.
- First `32` PCA components are used for affinity.
- Current row product vector is compared against completed-history centroids.
- Tail centroid threshold: train positive target p80.
- Low centroid threshold: train positive target p30.
- Historical rows only count if `offsaledate < current onsaledate`.

Affinity contexts:

- `global_affinity`
- `store_affinity`
- `store_segment_affinity`
- `store_subsegment_affinity`
- `store_title_affinity`
- `chain_title_affinity`
- `class_title_affinity`
- `chain_segment_affinity`
- `chain_class_subsegment_affinity`
- `class_subsegment_affinity`
- `chain_affinity`
- `class_affinity`
- `segment_affinity`
- `subsegment_affinity`

For each context, base affinity features:

- `*_tail_similarity`
- `*_low_similarity`
- `*_tail_minus_low_similarity`
- `*_tail_obs`
- `*_low_obs`
- `*_tail_avg_sales`
- `*_low_avg_sales`

For each context, derived affinity features:

- `*_tail_signal_x_analog_p90`
- `*_tail_signal_x_prior_avg`
- `*_tail_confidence`
- `*_tail_avg_over_prior_avg`
- `*_tail_avg_minus_low_avg`
- `*_tail_obs_log`

These features are the direct implementation of:

- Store embedding affinity.
- Contextual analog priors.
- Tail centroid similarity.
- Retrieval-style row analog signal through product similarity plus store/channel context.

## Training Changes

The regressor objective was updated to support stronger asymmetric decile pressure:

- Stronger bottom-overprediction penalty.
- Stronger top-underprediction penalty.
- Configurable top-under start decile.
- Total unit ratio penalty support.
- Stockout-aware gradient weighting support.
- Composite decile score written into run summaries.
- Configurable XGBoost depth, child weight, eta, subsample, and colsample.

The best base run still came from the older calibrated asymmetric run:

```text
new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_000228
```

The affinity-enhanced base regressor was tested:

```text
new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_003518
```

It did not improve the base-regressor result. Test composite worsened to `2.7928`. The conclusion is important: the semantic affinity features are better used for tail ranking/calibration than as direct unit regressors in the base model.

## Tail-Layer Process

The winning process:

1. Keep the old better-calibrated base regressor.
2. Train a positive-only tail classifier on the rebuilt feature set.
3. Target high-positive sales within positive rows, not positive-vs-zero incidence.
4. Use rank of tail probability as a monotonic multiplier signal.
5. Select multiplier by the composite decile metric on validation.
6. Avoid group-aware title/class calibration because it overfits.

Winning tail command shape:

```text
python3 new/train_tail_layer.py --family Weeklies --base-run-dir new/output/model_runs/weeklies/regressor_positive/asym_curve_log1p/20260517_000228 --tail-quantile 0.70 --rounds 350 --early-stopping 30 --multiplier-mode rank --calibration-group-cols '' --alpha-grid 1.8,2.0,2.2,2.4,2.6,2.8,3.0 --scale-grid 0.45,0.475,0.50,0.525,0.55,0.575,0.60
```

The selected validation multiplier was:

```text
alpha=2.4
scale=0.50
```

## Feature Importance Evidence

The tail classifier uses the new affinity features. High-gain affinity features included:

- `chain_class_subsegment_affinity_tail_avg_over_prior_avg`
- `store_title_affinity_tail_avg_sales`
- `store_title_affinity_tail_avg_over_prior_avg`
- `class_title_affinity_tail_obs`
- `store_segment_affinity_tail_avg_sales`
- `store_subsegment_affinity_tail_obs_log`
- `class_subsegment_affinity_tail_avg_over_prior_avg`
- `store_affinity_tail_confidence`
- `store_subsegment_affinity_tail_signal_x_analog_p90`

This confirms the feature set is not dead weight. The tail classifier is using store/title/channel affinity to rank high-positive rows.

## Negative Findings

The following did not become the winning approach:

- Affinity-enhanced base regressor alone.
- q60 tail classifier with group calibration.
- q70 tail classifier with `title,classoftrade` group calibration.
- Joint calibration table by base-prediction rank and tail-probability rank.
- Saturating/capping tail rank signal.
- Blending tail-probability rank with base-prediction rank.

The consistent failure mode of group and table calibration was validation overfit and weaker test shape.

## Current Recommendation

Keep the best current artifact as the Weeklies positive-only production candidate:

```text
new/output/tail_layer_runs/weeklies/regressor_positive/tail_q70/20260517_005103
```

Treat it as a real improvement over the base, but not final-final:

- It fixes most bottom overprediction.
- It fixes D10 starvation.
- It still underpredicts D7-D8.

The next modeling target should be a stricter selection metric that penalizes worst-case D7-D10 underprediction, not only mean D7-D10 underprediction. The current metric can allow D9-D10 overprediction to hide D7-D8 weakness.

