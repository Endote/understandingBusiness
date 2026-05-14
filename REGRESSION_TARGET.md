# Regression Target Design

## Business Reality

The client wants two things:
- forecast likely sales for each July `store_id x product_id`
- recommend a profit-aware `DrawQty`

So the regression target should represent forward demand as closely as possible, not accounting noise or allocation artifacts.

## Source Data Problems That Matter

### 1. Negative `SoldQty`

Historical `core.fact_sale` contains negative `SoldQty` rows.

Those rows are almost certainly:
- returns
- corrections
- stock/accounting adjustments

They are not meaningful future customer demand at the July issue-store prediction grain.

### 2. Censoring by `DrawQty`

Observed `SoldQty` can be capped by supply.

When:
- `drawqty > 0`
- `soldqty == drawqty`

then true demand may have been higher than observed sales.

So raw `SoldQty` is an observed-sales target, not a perfect demand target.

## Candidate Targets

### A. Raw `SoldQty`

Pros:
- simplest
- matches source directly

Cons:
- includes negative accounting adjustments
- mixes demand with supply constraints

Use:
- diagnostics only

### B. Clipped Sales Target: `sales_target = pmax(SoldQty, 0)`

Pros:
- removes clearly non-demand negative values
- stays in unit space the business cares about
- still aligned with draw optimization
- practical for first-pass modeling

Cons:
- still partially censored when `soldqty == drawqty`

Use:
- recommended primary regression target

### C. Sell-Through Target: `SoldQty / DrawQty`

Pros:
- normalizes by supply
- useful for allocation diagnostics

Cons:
- undefined or unstable when `drawqty == 0`
- still censored at stockout
- too dependent on historical draw policy

Use:
- secondary diagnostic target, not primary target

## Recommended Target

Primary training target:

```text
sales_target = max(SoldQty, 0)
```

Why:
- future customer demand cannot be negative
- negative source rows are likely accounting artifacts rather than demand
- the business decision ultimately needs expected positive unit sales

## Additional Modeling Flags

These should be carried as features or diagnostics:

- `negative_sales_flag = (SoldQty < 0)`
- `stockout_proxy_flag = (DrawQty > 0 and SoldQty == DrawQty)`
- `oversupply_units = DrawQty - pmax(SoldQty, 0)`
- `sellthrough = SoldQty / DrawQty` when `DrawQty > 0`

This lets the model preserve operational information without making the regression target itself incoherent.

## Recommended Modeling Setup

### Primary model

- target: `sales_target = pmax(SoldQty, 0)`
- family-specific models for `Weeklies` and `SIP`

### Secondary diagnostics

Track model performance separately for:
- all rows
- likely stockout rows
- negative-sales historical rows
- sparse stores / sparse product history

## Future Refinement

If the first-pass model is strong enough, the next upgrade should be to move from pure point regression toward demand-distribution modeling:
- quantile regression
- stockout-aware training
- possibly two-stage modeling:
  - probability of positive sales
  - expected units conditional on positive sales

But for the current business case, the correct first target is still:

```text
sales_target = max(SoldQty, 0)
```
