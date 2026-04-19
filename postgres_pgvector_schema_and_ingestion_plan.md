# Postgres + pgvector Schema And Ingestion Plan

Goal:
- Load the business-case data into Postgres cleanly
- Preserve the original relational grains
- Keep train and July holdout separated
- Store tabular features and embeddings in one database
- Stop after raw ingest and structural normalization, before resolving `DEMOGRAPHICS`

## Design Principles

- Use a minimal layered warehouse layout:
  - `raw`: exact file-level landing zone, bronze tier
  - `core`: conformed relational layer for modeling, silver tier
  - `holdout`: July pilot / holdout universe, kept separate from train
- Do not merge train and holdout universes.
- Do not denormalize dimensions into facts permanently.
- Keep embeddings in dedicated tables using `pgvector`.
- Delay `DEMOGRAPHICS` key enforcement until we decide the `POPULATION` resolution rule.
- Use scripts and ad hoc SQL checks for validation instead of persistent audit tables.
- Do not create persistent feature marts until feature engineering is settled.

## Database Setup

Required extension:

```sql
create extension if not exists vector;
```

Schemas:

```sql
create schema if not exists raw;
create schema if not exists core;
create schema if not exists holdout;
```

## Raw Layer

Purpose:
- Mirror source files exactly enough to support traceability and reprocessing
- Avoid irreversible transformations at ingest time
- Add only lightweight provenance fields when useful

### Train Raw Tables

```sql
create table raw.fact_sales (
    product_id bigint not null,
    store_id bigint not null,
    soldqty integer not null,
    drawqty integer not null
);

create table raw.dim_product (
    product_id bigint not null,
    barcode bigint,
    title text,
    type text,
    onsaledate date,
    offsaledate date,
    price numeric(12,4),
    segment text,
    subsegment text,
    frequency text
);

create table raw.dim_store (
    store_id bigint not null,
    postal_code text,
    store_chain text,
    region text,
    classoftrade text,
    merchandised text,
    facings integer,
    pockets integer
);

create table raw.demographics (
    postal_code text not null,
    age_19_and_under double precision,
    age_20_to_29 double precision,
    age_30_to_44 double precision,
    age_45_to_59 double precision,
    age_60_and_over double precision,
    male double precision,
    female double precision,
    two_or_more_races double precision,
    less_than_10k double precision,
    between_10k_and_14k double precision,
    between_15k_and_24k double precision,
    between_25k_and_34k double precision,
    between_35k_and_49k double precision,
    between_50k_and_74k double precision,
    between_75k_and_99k double precision,
    between_100k_and_149k double precision,
    between_150k_and_199k double precision,
    income_200k_or_more double precision,
    less_than_9th_grade double precision,
    between_9th_and_12th_grade_no_diploma double precision,
    high_school_graduate_includes_equivalency double precision,
    some_college_no_degree double precision,
    associates_degree double precision,
    bachelors_degree double precision,
    graduate_or_professional_degree double precision,
    population double precision,
    household_type_married_couple_household double precision,
    household_type_cohabiting_couple_household double precision,
    household_type_male_householder_no_spouse_partner_present double precision,
    household_type_female_householder_no_spouse_partner_present double precision,
    family_household double precision,
    non_family_household double precision
);

create table raw.contents_emb (
    product_id bigint not null,
    embedding_text text not null
);
```

### Holdout Raw Tables

```sql
create table raw.printing_schedule_holdout (
    store_id bigint not null,
    product_id bigint not null
);

create table raw.dim_product_holdout (
    product_id bigint not null,
    barcode bigint,
    title text,
    type text,
    onsaledate date,
    offsaledate date,
    price numeric(12,4),
    segment text,
    subsegment text,
    frequency text
);

create table raw.dim_store_holdout (
    store_id bigint not null,
    postal_code text,
    store_chain text,
    region text,
    classoftrade text,
    merchandised text,
    facings integer,
    pockets integer
);

create table raw.demographics_holdout (
    postal_code text not null,
    age_19_and_under double precision,
    age_20_to_29 double precision,
    age_30_to_44 double precision,
    age_45_to_59 double precision,
    age_60_and_over double precision,
    male double precision,
    female double precision,
    two_or_more_races double precision,
    less_than_10k double precision,
    between_10k_and_14k double precision,
    between_15k_and_24k double precision,
    between_25k_and_34k double precision,
    between_35k_and_49k double precision,
    between_50k_and_74k double precision,
    between_75k_and_99k double precision,
    between_100k_and_149k double precision,
    between_150k_and_199k double precision,
    income_200k_or_more double precision,
    less_than_9th_grade double precision,
    between_9th_and_12th_grade_no_diploma double precision,
    high_school_graduate_includes_equivalency double precision,
    some_college_no_degree double precision,
    associates_degree double precision,
    bachelors_degree double precision,
    graduate_or_professional_degree double precision,
    population double precision,
    household_type_married_couple_household double precision,
    household_type_cohabiting_couple_household double precision,
    household_type_male_householder_no_spouse_partner_present double precision,
    household_type_female_householder_no_spouse_partner_present double precision,
    family_household double precision,
    non_family_household double precision
);

create table raw.contents_emb_holdout (
    product_id bigint not null,
    embedding_text text not null
);
```

## Conformed Layer

Purpose:
- Enforce grains for entities already proven clean
- Convert embeddings into `vector`
- Keep `DEMOGRAPHICS` out of the conformed keyed layer until dedup logic is agreed

### Train Core Tables

```sql
create table core.fact_sales (
    product_id bigint not null,
    store_id bigint not null,
    soldqty integer not null,
    drawqty integer not null,
    primary key (product_id, store_id),
    check (soldqty >= 0),
    check (drawqty >= 0),
    check (soldqty <= drawqty)
);

create table core.dim_product (
    product_id bigint primary key,
    barcode bigint,
    title text,
    type text not null,
    onsaledate date not null,
    offsaledate date not null,
    price numeric(12,4),
    segment text,
    subsegment text,
    frequency text,
    check (offsaledate >= onsaledate)
);

create table core.dim_store (
    store_id bigint primary key,
    postal_code text,
    store_chain text,
    region text,
    classoftrade text,
    merchandised boolean,
    facings integer,
    pockets integer,
    check (facings >= 0),
    check (pockets >= 0)
);

create table core.dim_contents_emb (
    product_id bigint primary key,
    embedding vector(384)
);
```

Foreign keys to add after load:

```sql
alter table core.fact_sales
    add constraint fact_sales_product_fk
    foreign key (product_id) references core.dim_product(product_id);

alter table core.fact_sales
    add constraint fact_sales_store_fk
    foreign key (store_id) references core.dim_store(store_id);

alter table core.dim_contents_emb
    add constraint dim_contents_emb_product_fk
    foreign key (product_id) references core.dim_product(product_id);
```

### Holdout Conformed Tables

```sql
create table holdout.printing_schedule (
    store_id bigint not null,
    product_id bigint not null,
    primary key (store_id, product_id)
);

create table holdout.dim_product (
    product_id bigint primary key,
    barcode bigint,
    title text,
    type text not null,
    onsaledate date not null,
    offsaledate date not null,
    price numeric(12,4),
    segment text,
    subsegment text,
    frequency text,
    check (offsaledate >= onsaledate)
);

create table holdout.dim_store (
    store_id bigint primary key,
    postal_code text,
    store_chain text,
    region text,
    classoftrade text,
    merchandised boolean,
    facings integer,
    pockets integer,
    check (facings >= 0),
    check (pockets >= 0)
);

create table holdout.dim_contents_emb (
    product_id bigint primary key,
    embedding vector(384)
);
```

Foreign keys:

```sql
alter table holdout.printing_schedule
    add constraint printing_schedule_product_fk
    foreign key (product_id) references holdout.dim_product(product_id);

alter table holdout.printing_schedule
    add constraint printing_schedule_store_fk
    foreign key (store_id) references holdout.dim_store(store_id);

alter table holdout.dim_contents_emb
    add constraint holdout_dim_contents_emb_product_fk
    foreign key (product_id) references holdout.dim_product(product_id);
```

## DEMOGRAPHICS Staging Strategy

We are explicitly not resolving `DEMOGRAPHICS` yet.

So the near-term design is:
- load train and holdout demographics into `raw`
- do not promote either one into keyed `core` or `holdout` dimension tables yet
- inspect duplicates and decide the canonicalization rule outside the persistent warehouse design

And we stop there until we agree the canonicalization rule.

That is the correct pause point:
- raw tables loaded
- clean entities promoted to keyed layers
- `DEMOGRAPHICS` still only in raw tables

## Embedding Handling

The raw embedding files contain string representations of vectors.

Recommended ingestion pattern:
- land raw string in `raw.contents_emb.embedding_text`
- parse to numeric array in the ETL step
- cast into `vector(384)` in `core.dim_contents_emb` and `holdout.dim_contents_emb`

Example shape check:

```sql
select product_id
from raw.contents_emb
limit 10;
```

If the actual embedding dimension differs from `384`, adjust the `vector(n)` definition before final load. Do not hardcode blindly without one validation pass.

## Recommended Indexes

Core:

```sql
create index fact_sales_store_id_idx on core.fact_sales (store_id);
create index fact_sales_product_id_idx on core.fact_sales (product_id);
create index dim_product_type_idx on core.dim_product (type);
create index dim_product_onsaledate_idx on core.dim_product (onsaledate);
create index dim_store_postal_code_idx on core.dim_store (postal_code);
```

Holdout:

```sql
create index printing_schedule_store_id_idx on holdout.printing_schedule (store_id);
create index printing_schedule_product_id_idx on holdout.printing_schedule (product_id);
create index holdout_dim_store_postal_code_idx on holdout.dim_store (postal_code);
create index holdout_dim_product_onsaledate_idx on holdout.dim_product (onsaledate);
```

Optional vector similarity indexes, only when needed:

```sql
create index dim_contents_emb_embedding_ivfflat_idx
    on core.dim_contents_emb
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create index holdout_dim_contents_emb_embedding_ivfflat_idx
    on holdout.dim_contents_emb
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
```

Only create vector ANN indexes after the table is loaded and analyzed.

## Ingestion Plan

### Phase 1: Database Bootstrap

1. Create database.
2. Enable `pgvector`.
3. Create `raw`, `core`, `holdout` schemas.
4. Create raw tables.
5. Create conformed tables except any keyed `DEMOGRAPHICS` table.

### Phase 2: Raw File Load

Load each CSV exactly once into its matching raw table.

Suggested mapping:
- `input/FACT_TABLE.csv` -> `raw.fact_sales`
- `input/DIM_PRODUCT.csv` -> `raw.dim_product`
- `input/DIM_STORE.csv` -> `raw.dim_store`
- `input/DEMOGRAPHICS.csv` -> `raw.demographics`
- `input/EMBEDDINGS.csv` -> `raw.contents_emb`
- `input/Printing Schedule (On Sale July 2024)/Printing Schedule.csv` -> `raw.printing_schedule_holdout`
- `input/Printing Schedule (On Sale July 2024)/DIM_PRODUCT.csv` -> `raw.dim_product_holdout`
- `input/Printing Schedule (On Sale July 2024)/DIM_STORE.csv` -> `raw.dim_store_holdout`
- `input/Printing Schedule (On Sale July 2024)/DEMOGRAPHICS.csv` -> `raw.demographics_holdout`
- `input/Printing Schedule (On Sale July 2024)/EMBEDDINGS.csv` -> `raw.contents_emb_holdout`

Prefer `COPY` for performance.

### Phase 3: Raw Validation

Run these checks immediately after load:
- row counts per raw table
- PK duplicate counts against intended grains
- null checks on PK columns
- `soldqty <= drawqty`
- `offsaledate >= onsaledate`
- embedding parseability and dimension checks
- foreign-key coverage:
  - `raw.fact_sales.product_id` in `raw.dim_product`
  - `raw.fact_sales.store_id` in `raw.dim_store`
  - `raw.printing_schedule_holdout.product_id` in `raw.dim_product_holdout`
  - `raw.printing_schedule_holdout.store_id` in `raw.dim_store_holdout`

These validations should live as SQL scripts in the repo or as ad hoc queries, not as persistent warehouse tables.

### Phase 4: Promote Clean Entities

Promote all entities except `DEMOGRAPHICS`:
- `raw.fact_sales` -> `core.fact_sales`
- `raw.dim_product` -> `core.dim_product`
- `raw.dim_store` -> `core.dim_store`
- `raw.contents_emb` -> `core.dim_contents_emb`
- `raw.printing_schedule_holdout` -> `holdout.printing_schedule`
- `raw.dim_product_holdout` -> `holdout.dim_product`
- `raw.dim_store_holdout` -> `holdout.dim_store`
- `raw.contents_emb_holdout` -> `holdout.dim_contents_emb`

Transformations during promotion:
- convert `merchandised` from `Y/N` to boolean
- parse embeddings into `vector(n)`
- trim whitespace on free-text dimensions if needed

### Phase 5: Review Stop Point

At this point we stop and review `DEMOGRAPHICS`.

Deliverables available already:
- clean relational train layer
- clean relational holdout layer
- embeddings in `pgvector`
- unresolved `DEMOGRAPHICS` retained in raw only

That is enough to make the next decision deliberately instead of baking a bad demographic rule into the warehouse.

## Questions Explicitly Deferred

These should not be decided during raw ingest:
- how to canonicalize `DEMOGRAPHICS.POPULATION`
- whether to normalize malformed postal codes before or after demographic deduplication
- whether missing demographic postal codes should be imputed, dropped, or mapped
- whether train and holdout demographic cleanup should use the same canonicalization rule

## Recommended Next Step After This Plan

Implement:
- database bootstrap SQL
- raw ingest scripts
- validation SQL
- promotion SQL for all clean entities except `DEMOGRAPHICS`

Then review the demographic conflict patterns separately and decide the canonicalization rule with evidence.
