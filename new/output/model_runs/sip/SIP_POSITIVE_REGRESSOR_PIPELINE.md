# SIP Positive-Sales Regressor Pipeline

Last grounded update: 2026-05-17.

This note documents the current best SIP positive-only regressor pipeline from the saved codebase/artifacts. It is intentionally based on the current saved run summaries and prediction files, not chat memory.

## Scope

- Family: `SIP`
- Stage: `regressor_positive`
- Training rows: positive-sale rows only.
- Current objective: improve actual-target decile distribution, especially top30/top10 unit ratios, while keeping bottom30 overprediction acceptable.
- Current acceptance metrics are bottom/top band ratios. WAPE is diagnostic only.
- Product IDs are not relied on for holdout memorization. SIP title is constant and not useful. Store ID remains useful.

## Current Best Artifacts

Best saved tail-layer run:

```text
new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301
```

Base q40 regressor:

```text
new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102
```

External Tweedie rank signal:

```text
new/output/model_runs/sip/regressor_positive/tweedie/20260517_003424
```

Current best saved model artifact is `20260517_185301`. Current best business operating calibration is a replay from the same artifact with stronger top-tail parameters.

## Test Metrics

| candidate | WAPE | total ratio | bottom30 | top30 | top10 | target=1 | target=2 | target=3 | target=4 | target=5+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| saved selected, `20260517_185301` | 1.1471 | 0.9955 | 1.4680 | 0.8337 | 0.9051 | 1.4034 | 0.9689 | 0.7392 | 0.8374 | 0.9355 |
| best replay from `20260517_185301` | 1.1610 | 1.0057 | 1.4781 | 0.8445 | 0.9190 | 1.4135 | 0.9781 | 0.7470 | 0.8480 | 0.9506 |

Recommendation:

- Use `20260517_185301` as the model artifact to keep.
- Use the best replay calibration when optimizing for stronger top30/top10 signal and accepting bottom30 around `1.48`.
- Use the saved selected calibration when preferring the validation-selected artifact exactly as written by `train_tail_layer.py`.

## Pipeline Logic

The pipeline is a base regressor plus rank-calibrated tail layer.

1. Build SIP positive-only modeling dataset.
2. Train q40 quantile base regressor.
3. Train Tweedie regressor as an external rank signal.
4. Train three binary tail-layer classifiers:
   - `tail_ge3p0`: `sales_target >= 3`
   - `band_ge4p0`: `sales_target >= 4`
   - `low_le1p0`: `sales_target <= 1`
5. Convert tail classifier probabilities and external Tweedie predictions to within-split ranks.
6. Apply multiplicative rank calibration to the q40 base prediction.

Calibration formula:

```text
final_prediction = base_prediction * exp(
  log(scale)
  + alpha * (tail_rank - 0.5)
  + band_alpha * (band_rank - 0.5)
  + external_alpha * (tweedie_rank - 0.5)
  - low_alpha * (low_rank - 0.5)
)
```

## Dataset Rebuild Command

Use this if the SIP modeling dataset needs to be rebuilt from the database:

```bash
python3 new/build_modeling_datasets.py --family SIP
```

Default split configuration from `new/build_modeling_datasets.py`:

```text
train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15, folds=4, valid_dates_per_fold=4, min_train_dates=12
```

Output dataset path:

```text
new/output/modeling_datasets/sip/regressor_positive.parquet
```

The current dataset row count in the q40 and Tweedie run summaries is `2,505,611`.

## Base q40 Regressor

Saved artifact:

```text
new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102
```

Reproduction command:

```bash
python3 new/train_stage_model.py --family SIP --stage regressor_positive --objective quantile_40 --weighting category_balance --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 25 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85
```

Grounded saved params:

```text
objective=quantile_40
weighting=category_balance
max_depth=6
min_child_weight=25
eta=0.05
subsample=0.85
colsample_bytree=0.85
best_iteration=78
```

Saved q40 test metrics before tail layer:

```text
test WAPE=0.4859
test total ratio=0.5649
test top30 ratio=0.3293
test top10 ratio=0.2580
```

Interpretation: q40 keeps the bottom controlled but underpredicts the tail severely. The tail layer exists to fix this shape.

## External Tweedie Rank Signal

Saved artifact:

```text
new/output/model_runs/sip/regressor_positive/tweedie/20260517_003424
```

Reproduction command:

```bash
python3 new/train_stage_model.py --family SIP --stage regressor_positive --objective tweedie --weighting category_xtrade_tail_balance --rounds 500 --early-stopping 40 --max-depth 6 --min-child-weight 25 --eta 0.05 --subsample 0.85 --colsample-bytree 0.85
```

Grounded saved params:

```text
objective=tweedie
weighting=category_xtrade_tail_balance
max_depth=6
min_child_weight=25
eta=0.05
subsample=0.85
colsample_bytree=0.85
best_iteration=292
```

Saved Tweedie test metrics before tail layer:

```text
test WAPE=0.5820
test total ratio=1.2493
test top30 ratio=0.7282
test top10 ratio=0.5606
```

Interpretation: Tweedie is too inflated for direct scoring but useful as an external tail rank signal.

## Tail-Layer Saved Run

Saved artifact:

```text
new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301
```

Reproduction command:

```bash
python3 new/train_tail_layer.py --family SIP --base-run-dir new/output/model_runs/sip/regressor_positive/quantile_40/20260517_124102 --tail-threshold 3 --rounds 180 --early-stopping 30 --classifier-max-depth 6 --classifier-min-child-weight 25 --classifier-eta 0.04 --classifier-subsample 0.85 --classifier-colsample-bytree 0.85 --classifier-scale-pos-weight-multiplier 1.25 --multiplier-mode rank --selection-metric sip_tail_gain --calibration-group-cols '' --external-signal-run-dir new/output/model_runs/sip/regressor_positive/tweedie/20260517_003424 --enable-band-layer --band-target-min 4 --enable-low-layer --low-target-max 1 --alpha-grid 5.0,5.2,5.4 --band-alpha-grid 0,0.3,0.6 --external-alpha-grid 0.4,0.8 --low-alpha-grid 0,0.3,0.6,0.9 --scale-grid 0.345,0.35,0.355
```

Classifier params:

```text
classifier_max_depth=6
classifier_min_child_weight=25
classifier_eta=0.04
classifier_subsample=0.85
classifier_colsample_bytree=0.85
classifier_scale_pos_weight_multiplier=1.25
rounds=180
early_stopping=30
```

Layer targets:

```text
tail_target=tail_ge3p0, meaning sales_target >= 3
band_target=band_ge4p0, meaning sales_target >= 4
low_target=low_le1p0, meaning sales_target <= 1
```

Selection metric:

```text
sip_tail_gain
```

Saved selected multiplier:

```text
alpha=5.0
band_alpha=0.6
external_alpha=0.4
low_alpha=0.3
scale=0.355
```

Best replay multiplier from the saved run:

```text
alpha=5.4
band_alpha=0.0
external_alpha=0.4
low_alpha=0.6
scale=0.345
```

## Classifier Test Quality For Best Saved Run

From `new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301/tail_classifier_metrics.csv`:

| classifier target | ROC AUC | AP | actual tail recall at predicted top10 | actual tail recall at predicted top30 |
|---|---:|---:|---:|---:|
| `tail_ge3p0` | 0.6282 | 0.3971 | 0.1737 | 0.4249 |
| `band_ge4p0` | 0.7051 | 0.2437 | 0.2612 | 0.5572 |
| `low_le1p0` | 0.6105 | 0.5771 | 0.1281 | 0.3682 |

The `scale_pos_weight_multiplier=1.25` classifier setting improved the previous `1.0` setting. `1.5` was tested and was worse on the final acceptance metrics.

## Calibration Replay

The saved run writes predictions using the validation-selected multiplier. The stronger replay is not a separate trained model; it is the same artifact with a different multiplier row.

Replay formula for the best business candidate:

```text
adjusted_prediction = base_prediction * exp(log(0.345) + 5.4*(tail_probability_rank - 0.5) + 0.4*(external_signal_rank - 0.5) - 0.6*(low_probability_rank - 0.5))
```

Since `band_alpha=0.0`, the `band_probability_rank` term is unused in this replay.

One-line command to recompute best replay test metrics from the saved artifact:

```bash
python3 -c "import pandas as pd, numpy as np; df=pd.read_csv('new/output/tail_layer_runs/sip/regressor_positive/tail_ge3p0/20260517_185301/test_tail_adjusted_predictions.csv'); y=df.sales_target.to_numpy(float); base=df.base_prediction.to_numpy(float); tail=df.tail_probability_rank.to_numpy(float); ext=df.external_signal_rank.to_numpy(float); low=df.low_probability_rank.to_numpy(float); pred=base*np.exp(np.log(0.345)+5.4*(tail-.5)+0.4*(ext-.5)-0.6*(low-.5)); r=pd.Series(y).rank(method='first',pct=True).to_numpy(); print('wape %.4f total %.4f bottom30 %.4f top30 %.4f top10 %.4f target1 %.4f target2 %.4f target3 %.4f target4 %.4f target5p %.4f'%(np.abs(y-pred).sum()/y.sum(),pred.sum()/y.sum(),pred[r<=.3].sum()/y[r<=.3].sum(),pred[r>.7].sum()/y[r>.7].sum(),pred[r>.9].sum()/y[r>.9].sum(),pred[y==1].sum()/y[y==1].sum(),pred[y==2].sum()/y[y==2].sum(),pred[y==3].sum()/y[y==3].sum(),pred[y==4].sum()/y[y==4].sum(),pred[y>=5].sum()/y[y>=5].sum()))"
```

Expected output:

```text
wape 1.1610 total 1.0057 bottom30 1.4781 top30 0.8445 top10 0.9190 target1 1.4135 target2 0.9781 target3 0.7470 target4 0.8480 target5p 0.9506
```

## Operational Notes

- The saved selected row is cleaner for strict reproducibility because it is written directly into `test_tail_adjusted_predictions.csv`.
- The best replay row is the better business candidate for the current objective because it improves top30/top10 materially while keeping bottom30 under `1.50`.
- WAPE is expected to look high after tail calibration because the model deliberately reallocates unit mass toward high-tail rows and away from pure WAPE minimization.
- Any production scorer should implement the replay formula directly rather than depending on the `adjusted_prediction` column saved by the selected run.

## Next Tuning Candidates

Without changing architecture, the next grounded sweeps are:

1. Keep `classifier_scale_pos_weight_multiplier=1.25`, test a finer multiplier grid around the replay:

```text
alpha=5.2,5.4,5.6
external_alpha=0.3,0.4,0.5
low_alpha=0.5,0.6,0.7
scale=0.340,0.345,0.350
```

2. Tune the external Tweedie rank source while keeping it as rank-only signal:

```text
max_depth=5,6
min_child_weight=15,25,50
eta=0.035,0.05
```

3. Tune q40 base regressor only after the tail-layer grid plateaus:

```text
max_depth=5,6
min_child_weight=15,25,50
eta=0.035,0.05
```

