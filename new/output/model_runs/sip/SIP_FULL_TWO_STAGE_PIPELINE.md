# SIP Full Two-Stage Pipeline

Generated: 2026-05-17

This document records the current SIP production-style pipeline from the full SIP row universe:

1. all-row positive-sale classifier,
2. calibrated incidence probability,
3. positive-only amount regressor,
4. SIP tail layer using a SIP-specific Tweedie external signal,
5. all-row incidence x amount variant grid.

The results below are grounded in the current code and artifacts produced in this workspace, not in prior notes.

## Current Artifacts

- All-row SIP dataset: `new/output/modeling_datasets/sip/classifier_all.parquet`
- Incidence classifier: `new/output/model_runs/sip/classifier_all/binary_logistic/20260517_211635`
- Incidence calibration: `new/output/model_runs/sip/classifier_all/calibration/20260517_212917`
- Positive amount base regressor: `new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102`
- SIP Tweedie external signal: `new/output/model_runs/sip/regressor_positive/tweedie/20260517_003424`
- SIP tail layer: `new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301`
- Full two-stage grid: `new/output/pipeline_runs/sip/incidence_grid/20260517_215315`

## Reproduction Commands

Build the full SIP incidence dataset:

```bash
python3 new/build_incidence_dataset.py --family SIP
```

Train the all-row SIP incidence classifier. The full default feature set was too large for the local memory envelope, so this uses the full row sample but prunes high-volume embedding and amount-observation feature families:

```bash
python3 new/train_incidence_classifier.py --family SIP --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 50 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85 --drop-feature-substrings affinity,embedding_pca_,embedding_analog,_avg_sales,_obs --skip-train-metrics
```

Calibrate incidence probabilities:

```bash
python3 new/calibrate_incidence_classifier.py --family SIP --classifier-run-dir new/output/model_runs/sip/classifier_all/binary_logistic/20260517_211635 --target-overprediction 1.20 --group-shrinkage 25000 --group-min-rows 5000 --probability-bins 10
```

Train or recreate the SIP-specific external Tweedie signal. This must be SIP, not Weeklies:

```bash
python3 new/train_stage_model.py --family SIP --stage regressor_positive --objective tweedie --weighting category_xtrade_tail_balance --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 25 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85
```

Evaluate the full two-stage SIP pipeline using the current best positive-regressor tail multiplier:

```bash
python3 new/evaluate_incidence_pipeline.py --family SIP --base-run-dir new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102 --tail-run-dir new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301 --calibration-run-dir new/output/model_runs/sip/classifier_all/calibration/20260517_212917 --tail-alpha 5.4 --tail-band-alpha 0.0 --tail-external-alpha 0.4 --tail-low-alpha 0.6 --tail-scale 0.345
```

## Dataset Facts

The full SIP classifier dataset has 10,870,298 rows:

- train rows: 7,936,804
- validation rows: 1,294,369
- test rows: 1,639,125
- overall positive rate: 0.2305
- train positive rate: 0.2180
- validation positive rate: 0.2746
- test positive rate: 0.2564
- sales target median: 0
- sales target p90: 2
- sales target p99: 5

## Incidence Classifier

The full-sample lean classifier used 113 numeric features and 8 categorical features.

Validation training stopped around iteration 103:

- validation AUCPR: about 0.4366
- validation AUC: about 0.673

Selected calibration variant:

- `group_probability_bin_target_1.20`

Calibration test metrics:

| metric | value |
|---|---:|
| actual positive rate | 0.2564 |
| calibrated probability mean | 0.3137 |
| mean pred/actual decile ratio | 1.2405 |
| min pred/actual decile ratio | 1.1480 |
| max pred/actual decile ratio | 1.3273 |
| ROC AUC | 0.6647 |
| average precision | 0.4057 |
| Brier | 0.1814 |

## Amount Regressor And Tail Layer

The current positive-only amount stack is:

- q40 base amount regressor,
- broad tail classifier target `sales_target >= 3`,
- band and low layers loaded from the tail run,
- SIP-specific Tweedie external signal loaded from `new/output/model_runs/sip/regressor_positive/tweedie/20260517_003424`.

Tail multiplier used in the full pipeline grid:

| parameter | value |
|---|---:|
| alpha | 5.4 |
| band_alpha | 0.0 |
| external_alpha | 0.4 |
| low_alpha | 0.6 |
| scale | 0.345 |

## Full Two-Stage Test Results

The validation-selected production variant was:

`G_soft_gate_amount_raw_t0.70_floor0.00_low0.85_mid1.10_top0.95_scale0.95`

Test metrics:

| metric | value |
|---|---:|
| WAPE | 1.7114 |
| total unit ratio | 1.0656 |
| zero-row leak ratio | 0.4888 |
| bottom30 positive-decile ratio | 0.7759 |
| top30 positive-decile ratio | 0.5014 |
| top10 positive-decile ratio | 0.6210 |
| target=1 ratio | 0.7767 |
| target=2 ratio | 0.5902 |
| target=3 ratio | 0.9606 |
| target=5+ ratio | 0.6210 |

High top-signal exploratory candidates:

| variant | WAPE | total ratio | zero leak | bottom30 | top30 | top10 |
|---|---:|---:|---:|---:|---:|---:|
| `E_soft_gate_raw_threshold_0.65_floor_0.10` | 2.3812 | 2.0764 | 1.0841 | 1.4522 | 0.8226 | 0.9382 |
| `E_soft_gate_raw_threshold_0.65_floor_0.15` | 2.4193 | 2.1539 | 1.1394 | 1.4918 | 0.8379 | 0.9499 |
| `E_soft_gate_raw_threshold_0.65_floor_0.20` | 2.4604 | 2.2314 | 1.1947 | 1.5314 | 0.8532 | 0.9616 |
| `G_soft_gate_amount_raw_t0.60_floor0.00_low0.85_mid1.10_top0.95_scale0.95` | 2.4999 | 2.3059 | 1.2462 | 1.5929 | 0.8582 | 0.9429 |

Oracle actual-positive gate:

| metric | value |
|---|---:|
| WAPE | 1.2914 |
| total unit ratio | 1.3919 |
| zero-row leak ratio | 0.0000 |
| bottom30 | 2.1652 |
| top30 | 1.0984 |
| top10 | 1.1486 |

## Current Interpretation

The SIP-specific Tweedie external signal is now correctly wired into the full SIP pipeline. No Weeklies external signal is used.

The current bottleneck is the incidence gate, not the positive-only regressor. The positive-only oracle gate shows the amount stack can recover the top tail, but it badly overpredicts the low positive rows. The learned all-row gate has the opposite behavior: conservative production variants control total units and zero-row leakage, but they underfeed the high positive deciles.

The next empirical tuning should focus on incidence ranking and gating, not another amount architecture change:

- tune incidence thresholds/floors against a SIP-specific objective that includes bottom30, top30, top10, and zero-row leak;
- train a higher-recall SIP incidence classifier with tail-aware sample weights for positive rows with `sales_target >= 3`;
- add validation selection constraints instead of the current Weeklies-style score, because SIP acceptance is not the same as Weeklies acceptance;
- only after the incidence gate improves, revisit group calibration. Group calibration currently improves probability-decile calibration but does not solve top positive recall.
