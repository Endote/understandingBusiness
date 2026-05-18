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

## Tail-Weighted Incidence Experiment

I added a controlled classifier training option for positive-row and tail-positive sample weights. This is not a new architecture; it only changes the incidence classifier training weights.

Tail-weighted command:

```bash
python3 new/train_incidence_classifier.py --family SIP --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 50 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85 --drop-feature-substrings affinity,embedding_pca_,embedding_analog,_avg_sales,_obs --positive-weight 1.0 --tail-positive-threshold 3.0 --tail-positive-weight 3.0 --skip-train-metrics
```

Artifacts:

- Tail-weighted classifier: `new/output/model_runs/sip/classifier_all/binary_logistic/20260517_220808`
- Tail-weighted calibration: `new/output/model_runs/sip/classifier_all/calibration/20260517_221530`
- Tail-weighted full two-stage grid: `new/output/pipeline_runs/sip/incidence_grid/20260517_222531`

Incidence test ranking comparison at predicted top 30%:

| run | AP | AUC | positive recall | tail>=3 recall | tail>=5 recall | sales units captured |
|---|---:|---:|---:|---:|---:|---:|
| base incidence | 0.4057 | 0.6647 | 0.4688 | 0.5842 | 0.7714 | 453,436 |
| tail-weighted incidence | 0.4064 | 0.6636 | 0.4681 | 0.5885 | 0.7825 | 455,975 |

Full two-stage test comparison:

| run | variant | WAPE | total ratio | zero leak | bottom30 | top30 | top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| base selected | `G_soft_gate_amount_raw_t0.70_floor0.00_low0.85_mid1.10_top0.95_scale0.95` | 1.7114 | 1.0656 | 0.4888 | 0.7759 | 0.5014 | 0.6210 |
| tail-weighted selected | `G_soft_gate_amount_calibrated_t0.55_floor0.00_low0.85_mid1.10_top0.95_scale0.95` | 1.6752 | 0.9876 | 0.4426 | 0.7014 | 0.4833 | 0.6199 |
| base high-top candidate | `E_soft_gate_raw_threshold_0.65_floor_0.15` | 2.4193 | 2.1539 | 1.1394 | 1.4918 | 0.8379 | 0.9499 |
| tail-weighted high-top candidate | `E_soft_gate_raw_threshold_0.65_floor_0.15` | 3.0208 | 2.9863 | 1.6992 | 1.9614 | 1.0322 | 1.1091 |

Conclusion: tail weighting improved the classifier's broad/high-tail ranking slightly, but the downstream high-recall gate explodes zero-row leakage and bottom positives. It is not a better deployable candidate yet. The useful signal is that the classifier can move tail recall, but the pipeline needs a SIP-specific selection objective and a better gating/calibration frontier rather than a blunt tail-weight multiplier.

## SIP-Specific Gate Selection Runs

I added a SIP-specific selection score and a denser gate grid in `new/evaluate_incidence_pipeline.py`.

The SIP score optimizes:

- zero-row leak control,
- total unit ratio not exploding,
- bottom positive deciles not exceeding the 1.2-1.4 overshoot range too much,
- top30 and top10 positive decile recovery,
- minimum positive decile ratio not collapsing.

Base-incidence dense gate run:

- Run: `new/output/pipeline_runs/sip/incidence_grid/20260517_230159`
- Classifier: `new/output/model_runs/sip/classifier_all/binary_logistic/20260517_211635`
- Calibration: `new/output/model_runs/sip/classifier_all/calibration/20260517_212917`

Best high-top constrained candidate from this run:

| variant | WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | min decile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SIP_dense_soft_raw_t0.675_floor0.025` | 2.0987 | 1.6136 | 0.7935 | 1.1554 | 0.6959 | 0.8264 | 1.4544 | 0.6128 |
| `SIP_dense_soft_raw_t0.675_floor0.000` | 2.0804 | 1.5660 | 0.7605 | 1.1295 | 0.6856 | 0.8182 | 1.4210 | 0.6021 |

Interpretation: this is the best top-tail frontier so far while keeping zero leak under 0.80. It slightly exceeds the 1.4 max-positive-decile cap, but it materially improves top10 and top30.

Avg-sales incidence dense gate run:

- Classifier: `new/output/model_runs/sip/classifier_all/binary_logistic/20260517_232319`
- Calibration: `new/output/model_runs/sip/classifier_all/calibration/20260517_233124`
- Full grid: `new/output/pipeline_runs/sip/incidence_grid/20260517_234120`
- Feature change: kept completed historical `_avg_sales` priors while still dropping `affinity`, `embedding_pca_`, `embedding_analog`, and `_obs` features.

Best strict overshoot-control candidates:

| variant | WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | min decile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SIP_dense_soft_raw_t0.700_floor0.025` | 1.9613 | 1.4065 | 0.6730 | 1.0025 | 0.6286 | 0.7588 | 1.3278 | 0.5547 |
| `SIP_dense_soft_raw_t0.700_floor0.050` | 1.9805 | 1.4595 | 0.7090 | 1.0323 | 0.6406 | 0.7688 | 1.3645 | 0.5668 |
| `SIP_dense_soft_raw_t0.700_floor0.075` | 2.0002 | 1.5124 | 0.7451 | 1.0621 | 0.6527 | 0.7788 | 1.4011 | 0.5790 |

Interpretation: restoring completed average-sales priors improved the production-control side of the frontier. The best strict candidate is `SIP_dense_soft_raw_t0.700_floor0.050`: all positive deciles stay below 1.4, zero leak is 0.7090, and total is 1.4595. It sacrifices some top-tail signal compared with the base-incidence `t0.675` candidates.

Current recommendation:

- If strict decile overshoot control is the priority, move forward with `avg_sales_dense` + `SIP_dense_soft_raw_t0.700_floor0.050`.
- If top-tail signal is still the priority and a small max-decile overshoot breach is acceptable, keep `base_dense` + `SIP_dense_soft_raw_t0.675_floor0.000` or `floor0.025` as the research frontier.

## Amount-Band, Zero-Veto, And Optimized Postprocessor

I added the requested production-style refinements on top of the SIP q40/tail/Tweedie amount stack:

- amount-band incidence model, trained as multiclass bands `zero`, `target=1`, `target=2`, `target=3-4`, `target=5+`;
- zero-leak veto model, trained only on the candidate predicted-positive region from the base incidence gate;
- optimized postprocessor, selecting a blended rank from base incidence, avg-sales incidence, amount-band expected units, tail probability, and zero-veto risk;
- avg-sales prior exploitation through a separate avg-sales incidence classifier blended into the postprocessor.

These models keep `product_id` out of the feature set and do not rely on unavailable scoring-time `drawqty`, `soldqty`, or stockout labels.

Amount-band incidence command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/train_amount_band_incidence.py --family SIP --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 50 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85 --drop-feature-substrings affinity,embedding_pca_,embedding_analog,_obs --skip-train-metrics
```

Amount-band artifact:

- `new/output/model_runs/sip/classifier_all/amount_band_multiclass/20260518_001723`

Amount-band test diagnostics:

| metric | value |
|---|---:|
| log loss | 0.8928 |
| actual mean units | 0.5214 |
| expected units mean | 0.9292 |
| positive probability mean | 0.4101 |
| tail probability mean | 0.1374 |
| top30 by expected-units sales captured | 453,706 |
| top30 by expected-units tail>=3 recall | 0.5893 |

Zero-leak veto command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/train_zero_leak_veto.py --family SIP --base-calibration-run-dir new/output/model_runs/sip/classifier_all/calibration/20260517_212917 --candidate-threshold 0.675 --rounds 500 --early-stopping 40 --max-depth 5 --min-child-weight 50 --eta 0.04 --subsample 0.85 --colsample-bytree 0.85 --drop-feature-substrings affinity,embedding_pca_,embedding_analog,_obs --skip-train-metrics
```

Zero-veto artifact:

- `new/output/model_runs/sip/classifier_all/zero_leak_veto/20260518_002153`

Veto validation diagnostics:

| metric | value |
|---|---:|
| candidate train rows | 738,626 |
| candidate validation rows | 201,004 |
| candidate train zero rate | 0.5259 |
| validation AUCPR | about 0.5610 |
| validation AUC | about 0.5753 |

Optimized postprocessor command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/optimize_sip_postprocessor.py --base-calibration-run-dir new/output/model_runs/sip/classifier_all/calibration/20260517_212917 --avg-calibration-run-dir new/output/model_runs/sip/classifier_all/calibration/20260517_233124 --amount-band-run-dir new/output/model_runs/sip/classifier_all/amount_band_multiclass/20260518_001723 --veto-run-dir new/output/model_runs/sip/classifier_all/zero_leak_veto/20260518_002153 --base-run-dir new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102 --tail-run-dir new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301 --tail-alpha 5.4 --tail-band-alpha 0.0 --tail-external-alpha 0.4 --tail-low-alpha 0.6 --tail-scale 0.345 --trials 2500 --top-k-test 160
```

Optimized postprocessor artifact:

- `new/output/pipeline_runs/sip/postprocessor_optimization/20260518_012557`

Validation-selected candidate:

| candidate | base | avg | band | tail | veto | threshold | floor | scale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 892 | 0.5296 | 0.0564 | 0.3672 | 0.0468 | 0.0123 | 0.9172 | 0.0000 | 1.0181 |

Validation metrics for candidate 892:

| WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | min decile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0197 | 1.4446 | 0.6343 | 1.0382 | 0.7173 | 0.8562 | 1.1423 | 0.6301 |

Test metrics for candidate 892:

| WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | min decile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0575 | 1.4762 | 0.7060 | 1.0464 | 0.6675 | 0.8179 | 1.3574 | 0.5823 |

Test positive-decile ratios for candidate 892:

| D1 | D2 | D3 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|
| 1.0029 | 0.7790 | 1.3574 | 0.6025 | 0.5823 | 0.8179 |

Best observed test frontier rows from the optimized postprocessor:

| candidate | WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | min decile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1102 | 2.0490 | 1.4627 | 0.6946 | 1.0475 | 0.6655 | 0.8161 | 1.3883 | 0.5784 |
| 892 | 2.0575 | 1.4762 | 0.7060 | 1.0464 | 0.6675 | 0.8179 | 1.3574 | 0.5823 |
| 1806 | 2.0470 | 1.4900 | 0.7122 | 1.0687 | 0.6696 | 0.8113 | 1.3932 | 0.5878 |
| 439 | 2.0691 | 1.4931 | 0.7100 | 1.0624 | 0.6772 | 0.8237 | 1.4082 | 0.5961 |
| 2358 | 2.1035 | 1.5370 | 0.7429 | 1.0774 | 0.6899 | 0.8494 | 1.4470 | 0.6023 |

Comparison against prior candidates:

| candidate | WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | min decile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base dense `t0.675_floor0.000` | 2.0804 | 1.5660 | 0.7605 | 1.1295 | 0.6856 | 0.8182 | 1.4210 | 0.6021 |
| avg-sales dense `t0.700_floor0.050` | 1.9805 | 1.4595 | 0.7090 | 1.0323 | 0.6406 | 0.7688 | 1.3645 | 0.5668 |
| optimized candidate 892 | 2.0575 | 1.4762 | 0.7060 | 1.0464 | 0.6675 | 0.8179 | 1.3574 | 0.5823 |
| optimized candidate 439 | 2.0691 | 1.4931 | 0.7100 | 1.0624 | 0.6772 | 0.8237 | 1.4082 | 0.5961 |

Current recommendation:

- Move forward with optimized candidate 892 as the best validation-selected controlled production baseline.
- Keep candidate 439 as the high-top research variant if a small max-decile breach above 1.40 is acceptable.
- The amount-band signal helped the most in the selected blend. The zero-veto model is currently weak, so it is useful as a small negative rank term but not strong enough to be a hard veto.
- The next improvement should target better incidence separation inside the predicted-positive region, because postprocessing now finds a reasonable tradeoff but cannot lift top30 much past about 0.67 without either raising max decile above 1.40 or increasing zero leak.

## Cumulative Amount And Lift-Layer Grid

I added a cumulative amount-incidence stack and a lift-layer optimizer anchored on validation-selected candidate 892.

New scripts:

- `new/train_cumulative_amount_incidence.py`
- `new/train_low_positive_veto.py`
- `new/optimize_sip_lift_layer.py`

Cumulative amount command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/train_cumulative_amount_incidence.py --family SIP --thresholds 1,2,3,4,5 --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 50 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85 --drop-feature-substrings affinity,embedding_pca_,embedding_analog,_obs
```

Cumulative artifact:

- `new/output/model_runs/sip/classifier_all/cumulative_amount_incidence/20260518_014236`

Cumulative test diagnostics:

| target | AUC | AP | recall at top30 | sales units at top30 |
|---:|---:|---:|---:|---:|
| sales>=1 | 0.6654 | 0.4060 | 0.4695 | 454,406 |
| sales>=2 | 0.6934 | 0.2654 | 0.5410 | 456,933 |
| sales>=3 | 0.7150 | 0.1834 | 0.5921 | 456,328 |
| sales>=4 | 0.7842 | 0.1269 | 0.7133 | 452,474 |
| sales>=5 | 0.8481 | 0.1041 | 0.8219 | 452,518 |

Low-positive veto command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/train_low_positive_veto.py --family SIP --incidence-run-dir new/output/model_runs/sip/classifier_all/binary_logistic/20260517_211635 --candidate-threshold 0.675 --label-mode le1 --rounds 500 --early-stopping 40 --max-depth 5 --min-child-weight 50 --eta 0.04 --subsample 0.85 --colsample-bytree 0.85 --drop-feature-substrings affinity,embedding_pca_,embedding_analog,_obs
```

Low-positive veto artifact:

- `new/output/model_runs/sip/classifier_all/low_positive_veto/20260518_021510`

Low-positive veto test diagnostics:

| label | candidate rows | candidate label rate | AUC | AP |
|---|---:|---:|---:|---:|
| sales<=1 | 182,682 | 0.6934 | 0.4863 | 0.8553 |

Interpretation: the low-positive veto is weak out of sample. It is acceptable as a damp rank, but not as a hard veto.

Strict lift-layer command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 new/optimize_sip_lift_layer.py --base-calibration-run-dir new/output/model_runs/sip/classifier_all/calibration/20260517_212917 --avg-calibration-run-dir new/output/model_runs/sip/classifier_all/calibration/20260517_233124 --amount-band-run-dir new/output/model_runs/sip/classifier_all/amount_band_multiclass/20260518_001723 --cumulative-run-dir new/output/model_runs/sip/classifier_all/cumulative_amount_incidence/20260518_014236 --veto-run-dir new/output/model_runs/sip/classifier_all/zero_leak_veto/20260518_002153 --low-veto-run-dir new/output/model_runs/sip/classifier_all/low_positive_veto/20260518_021510 --base-run-dir new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102 --tail-run-dir new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301 --tail-alpha 5.4 --tail-band-alpha 0.0 --tail-external-alpha 0.4 --tail-low-alpha 0.6 --tail-scale 0.345 --trials 20000 --screen-sample-rows 250000 --screen-top-k 1500 --top-k-test 600
```

Strict lift-layer artifact:

- `new/output/pipeline_runs/sip/lift_layer_optimization/20260518_024421`

Baseline 892 versus strict lift candidates on test:

| candidate | WAPE | total | zero leak | bottom30 | top30 | top10 | max decile | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base 892 | 2.0575 | 1.4762 | 0.7060 | 1.0464 | 0.6675 | 0.8179 | 1.3574 | 0.6025 | 0.5823 | 0.8179 |
| selected lift 12363 | 2.1432 | 1.5513 | 0.7164 | 1.0871 | 0.7411 | 0.9519 | 1.4612 | 0.6420 | 0.6293 | 0.9519 |
| strict frontier 11655 | 2.1379 | 1.5477 | 0.7183 | 1.0772 | 0.7365 | 0.9441 | 1.4350 | 0.6400 | 0.6253 | 0.9441 |
| strict frontier 7255 | 2.1287 | 1.5422 | 0.7161 | 1.0760 | 0.7325 | 0.9372 | 1.4282 | 0.6387 | 0.6217 | 0.9372 |

The lift layer confirms the cumulative amount signal is real: top30 moves from 0.6675 to the 0.73-0.74 range with zero leak still near 0.716-0.718 and bottom30 around 1.08. The cost is D3/max-decile pressure: the best top30 candidates sit around max decile 1.43-1.46 and total 1.54-1.55.

Current lift-layer recommendation:

- Keep candidate 892 as the safest production baseline.
- Promote strict frontier candidate 11655 as the best high-top controlled research candidate if max decile 1.435 and total 1.548 are acceptable.
- Do not treat the selected lift candidate 12363 as strictly superior; it gives slightly more top30 but breaches max decile above 1.45.
- The next modeling improvement should be a stronger low-positive separator, because cumulative `sales>=3/4/5` works, but the current low-positive veto fails to reliably distinguish D3-like one-unit rows from D8/D9 rows.
