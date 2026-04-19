# Primary Key Duplicate Audit

Scope:
- Training extract in `input/`
- Holdout / pilot extract in `input/Printing Schedule (On Sale July 2024)/`

Method:
- Validate each entity against its intended primary key from the case schema.
- For duplicated primary keys, classify them into:
  - `safe_dedup_groups`: duplicated PK groups where all non-key columns are identical
  - `conflicting_groups`: duplicated PK groups where at least one non-key column differs

Important note on `DEMOGRAPHICS`:
- All duplicated `POSTAL_CODE` groups in both train and holdout have identical demographic share columns except for `POPULATION`.
- Therefore, these are not conflicting demographic profiles, but they are still conflicting rows at the relational level and cannot be blindly deduplicated without an explicit `POPULATION` resolution rule.

## Summary Table

| Scope | Entity | Intended PK | Rows | Duplicate PK Rows | Duplicate PK Groups | Safe Dedup Groups | Conflicting Groups |
|---|---|---|---:|---:|---:|---:|---:|
| train | FACT_SALES | (`PRODUCT_ID`, `STORE_ID`) | 16,590,626 | 0 | 0 | 0 | 0 |
| train | DIM_PRODUCT | (`PRODUCT_ID`) | 5,324 | 0 | 0 | 0 | 0 |
| train | DIM_STORE | (`STORE_ID`) | 29,622 | 0 | 0 | 0 | 0 |
| train | DEMOGRAPHICS | (`POSTAL_CODE`) | 28,709 | 26,198 | 4,876 | 453 | 4,423 |
| train | CONTENTS_EMB | (`PRODUCT_ID`) | 5,723 | 0 | 0 | 0 | 0 |
| holdout | PRINTING_SCHEDULE | (`STORE_ID`, `PRODUCT_ID`) | 761,531 | 0 | 0 | 0 | 0 |
| holdout | DIM_PRODUCT | (`PRODUCT_ID`) | 298 | 0 | 0 | 0 | 0 |
| holdout | DIM_STORE | (`STORE_ID`) | 27,714 | 0 | 0 | 0 | 0 |
| holdout | DEMOGRAPHICS | (`POSTAL_CODE`) | 26,861 | 24,253 | 4,724 | 521 | 4,203 |
| holdout | CONTENTS_EMB | (`PRODUCT_ID`) | 317 | 0 | 0 | 0 | 0 |

## Interpretation

- Every entity except `DEMOGRAPHICS` respects the intended primary key perfectly in both extracts.
- `FACT_SALES` and `PRINTING_SCHEDULE` are clean at their composite key grain.
- `DIM_PRODUCT`, `DIM_STORE`, and `CONTENTS_EMB` are clean at their single-column key grain.
- The only entity that violates its intended relational grain is `DEMOGRAPHICS`.

## DEMOGRAPHICS Detail

Training extract:
- Duplicated `POSTAL_CODE` groups: 4,876
- Fully identical duplicate groups: 453
- Duplicate groups with differing non-key values: 4,423
- Maximum distinct non-key row variants per postal code: 3

Holdout extract:
- Duplicated `POSTAL_CODE` groups: 4,724
- Fully identical duplicate groups: 521
- Duplicate groups with differing non-key values: 4,203
- Maximum distinct non-key row variants per postal code: 3

Critical nuance:
- For all duplicated postal codes in both train and holdout, the demographic share columns are identical when `POPULATION` is excluded.
- The apparent row conflicts are driven entirely by differing `POPULATION` values.

Equivalent restatement:
- Train: all 4,876 duplicated postal codes are identical except for `POPULATION`
- Holdout: all 4,724 duplicated postal codes are identical except for `POPULATION`

## Examples of Conflicting DEMOGRAPHICS PK Groups

Examples from training extract:
- `24-678`: 5 rows, 3 distinct full-row variants
- `24-684`: 11 rows, 3 distinct full-row variants
- `24-685`: 2 rows, 2 distinct full-row variants
- `24-689`: 5 rows, 2 distinct full-row variants
- `24-690`: 2 rows, 2 distinct full-row variants

These are not different share profiles. They are repeated postal codes carrying different `POPULATION` values.

## Conclusion for Loading Design

- The source schema is relationally clean except for `DEMOGRAPHICS`.
- We should load raw files as-is into Postgres.
- We should enforce primary keys only in the conformed layer, not directly on raw ingest.
- `DEMOGRAPHICS` needs a canonicalization step to one row per `POSTAL_CODE` before joining into feature marts.
- That canonicalization needs an explicit rule for `POPULATION`; the demographic share columns themselves can be preserved as-is because they are stable within duplicated postal codes.
