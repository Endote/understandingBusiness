\timing on

\echo 'Creating extension and schemas'
create extension if not exists vector;

drop schema if exists raw cascade;
drop schema if exists core cascade;
drop schema if exists holdout cascade;

create schema raw;
create schema core;
create schema holdout;

\echo 'Creating raw tables'
create table raw.fact_sale (
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

create table raw.demographic (
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

create table raw.content_embedding (
    product_id bigint not null,
    embedding_text text not null
);

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

create table raw.demographic_holdout (
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

create table raw.content_embedding_holdout (
    product_id bigint not null,
    embedding_text text not null
);

\echo 'Loading raw CSV data'
\copy raw.fact_sale (store_id, product_id, soldqty, drawqty) from '__TRAIN_FACT_CSV__' with (format csv, header true)
\copy raw.dim_product from '__TRAIN_PRODUCT_CSV__' with (format csv, header true)
\copy raw.dim_store from '__TRAIN_STORE_CSV__' with (format csv, header true)
\copy raw.demographic from '__TRAIN_DEMOGRAPHIC_CSV__' with (format csv, header true)
\copy raw.content_embedding from '__TRAIN_EMBEDDING_CSV__' with (format csv, header true)
\copy raw.printing_schedule_holdout from '__HOLDOUT_SCHEDULE_CSV__' with (format csv, header true)
\copy raw.dim_product_holdout from '__HOLDOUT_PRODUCT_CSV__' with (format csv, header true)
\copy raw.dim_store_holdout from '__HOLDOUT_STORE_CSV__' with (format csv, header true)
\copy raw.demographic_holdout from '__HOLDOUT_DEMOGRAPHIC_CSV__' with (format csv, header true)
\copy raw.content_embedding_holdout from '__HOLDOUT_EMBEDDING_CSV__' with (format csv, header true)

\echo 'Running raw validations'
do $$
begin
    if exists (select 1 from raw.fact_sale where product_id is null or store_id is null) then
        raise exception 'raw.fact_sale has null primary key values';
    end if;
    if exists (
        select 1
        from raw.fact_sale
        group by product_id, store_id
        having count(*) > 1
    ) then
        raise exception 'raw.fact_sale has duplicate (product_id, store_id) rows';
    end if;
    if exists (select 1 from raw.fact_sale where drawqty < 0 or soldqty > drawqty) then
        raise exception 'raw.fact_sale violates drawqty>=0 or soldqty<=drawqty checks';
    end if;

    if exists (select 1 from raw.dim_product where product_id is null) then
        raise exception 'raw.dim_product has null product_id';
    end if;
    if exists (
        select 1
        from raw.dim_product
        group by product_id
        having count(*) > 1
    ) then
        raise exception 'raw.dim_product has duplicate product_id';
    end if;
    if exists (select 1 from raw.dim_product where offsaledate < onsaledate) then
        raise exception 'raw.dim_product has offsaledate before onsaledate';
    end if;

    if exists (select 1 from raw.dim_store where store_id is null) then
        raise exception 'raw.dim_store has null store_id';
    end if;
    if exists (
        select 1
        from raw.dim_store
        group by store_id
        having count(*) > 1
    ) then
        raise exception 'raw.dim_store has duplicate store_id';
    end if;

    if exists (select 1 from raw.content_embedding where product_id is null) then
        raise exception 'raw.content_embedding has null product_id';
    end if;
    if exists (
        select 1
        from raw.content_embedding
        group by product_id
        having count(*) > 1
    ) then
        raise exception 'raw.content_embedding has duplicate product_id';
    end if;

    if exists (select 1 from raw.printing_schedule_holdout where store_id is null or product_id is null) then
        raise exception 'raw.printing_schedule_holdout has null primary key values';
    end if;
    if exists (
        select 1
        from raw.printing_schedule_holdout
        group by store_id, product_id
        having count(*) > 1
    ) then
        raise exception 'raw.printing_schedule_holdout has duplicate (store_id, product_id) rows';
    end if;

    if exists (select 1 from raw.dim_product_holdout where product_id is null) then
        raise exception 'raw.dim_product_holdout has null product_id';
    end if;
    if exists (
        select 1
        from raw.dim_product_holdout
        group by product_id
        having count(*) > 1
    ) then
        raise exception 'raw.dim_product_holdout has duplicate product_id';
    end if;
    if exists (select 1 from raw.dim_product_holdout where offsaledate < onsaledate) then
        raise exception 'raw.dim_product_holdout has offsaledate before onsaledate';
    end if;

    if exists (select 1 from raw.dim_store_holdout where store_id is null) then
        raise exception 'raw.dim_store_holdout has null store_id';
    end if;
    if exists (
        select 1
        from raw.dim_store_holdout
        group by store_id
        having count(*) > 1
    ) then
        raise exception 'raw.dim_store_holdout has duplicate store_id';
    end if;

    if exists (select 1 from raw.content_embedding_holdout where product_id is null) then
        raise exception 'raw.content_embedding_holdout has null product_id';
    end if;
    if exists (
        select 1
        from raw.content_embedding_holdout
        group by product_id
        having count(*) > 1
    ) then
        raise exception 'raw.content_embedding_holdout has duplicate product_id';
    end if;

    if exists (
        select 1
        from raw.demographic d
        group by postal_code
        having count(distinct (to_jsonb(d) - 'population')) > 1
    ) then
        raise exception 'raw.demographic has conflicting non-population values within duplicated postal_code groups';
    end if;

    if exists (
        select 1
        from raw.demographic_holdout d
        group by postal_code
        having count(distinct (to_jsonb(d) - 'population')) > 1
    ) then
        raise exception 'raw.demographic_holdout has conflicting non-population values within duplicated postal_code groups';
    end if;
end $$;

\echo 'Validation summary'
select 'raw.fact_sale.rows' as metric, count(*)::bigint as value from raw.fact_sale;
select 'raw.dim_product.rows' as metric, count(*)::bigint as value from raw.dim_product;
select 'raw.dim_store.rows' as metric, count(*)::bigint as value from raw.dim_store;
select 'raw.demographic.rows' as metric, count(*)::bigint as value from raw.demographic;
select 'raw.content_embedding.rows' as metric, count(*)::bigint as value from raw.content_embedding;
select 'raw.printing_schedule_holdout.rows' as metric, count(*)::bigint as value from raw.printing_schedule_holdout;
select 'raw.dim_product_holdout.rows' as metric, count(*)::bigint as value from raw.dim_product_holdout;
select 'raw.dim_store_holdout.rows' as metric, count(*)::bigint as value from raw.dim_store_holdout;
select 'raw.demographic_holdout.rows' as metric, count(*)::bigint as value from raw.demographic_holdout;
select 'raw.content_embedding_holdout.rows' as metric, count(*)::bigint as value from raw.content_embedding_holdout;
select 'raw.demographic.duplicate_postal_code_groups' as metric, count(*)::bigint as value
from (
    select postal_code
    from raw.demographic
    group by postal_code
    having count(*) > 1
) s;
select 'raw.demographic_holdout.duplicate_postal_code_groups' as metric, count(*)::bigint as value
from (
    select postal_code
    from raw.demographic_holdout
    group by postal_code
    having count(*) > 1
) s;
select 'raw.fact_sale.product_fk_miss' as metric, count(*)::bigint as value
from raw.fact_sale fs
left join raw.dim_product dp on dp.product_id = fs.product_id
where dp.product_id is null;
select 'raw.fact_sale.negative_soldqty_rows' as metric, count(*)::bigint as value
from raw.fact_sale
where soldqty < 0;
select 'raw.fact_sale.store_fk_miss' as metric, count(*)::bigint as value
from raw.fact_sale fs
left join raw.dim_store ds on ds.store_id = fs.store_id
where ds.store_id is null;
select 'raw.printing_schedule_holdout.product_fk_miss' as metric, count(*)::bigint as value
from raw.printing_schedule_holdout ps
left join raw.dim_product_holdout dp on dp.product_id = ps.product_id
where dp.product_id is null;
select 'raw.printing_schedule_holdout.store_fk_miss' as metric, count(*)::bigint as value
from raw.printing_schedule_holdout ps
left join raw.dim_store_holdout ds on ds.store_id = ps.store_id
where ds.store_id is null;

\echo 'Preparing embedding parsing function'
create or replace function core.parse_embedding_text(embedding_text text)
returns vector(384)
language sql
immutable
strict
as $$
    select (
        '[' ||
        regexp_replace(
            trim(both ' ' from trim(both '[]' from regexp_replace(embedding_text, E'[\\n\\r]+', ' ', 'g'))),
            '\s+',
            ',',
            'g'
        ) ||
        ']'
    )::vector(384)
$$;

do $$
declare
    train_bad_count integer;
    holdout_bad_count integer;
begin
    select count(*)
    into train_bad_count
    from raw.content_embedding
    where array_length(
        regexp_split_to_array(
            trim(both ' ' from trim(both '[]' from regexp_replace(embedding_text, E'[\\n\\r]+', ' ', 'g'))),
            '\s+'
        ),
        1
    ) <> 384;

    select count(*)
    into holdout_bad_count
    from raw.content_embedding_holdout
    where array_length(
        regexp_split_to_array(
            trim(both ' ' from trim(both '[]' from regexp_replace(embedding_text, E'[\\n\\r]+', ' ', 'g'))),
            '\s+'
        ),
        1
    ) <> 384;

    if train_bad_count > 0 then
        raise exception 'raw.content_embedding contains % rows with non-384 dimensions', train_bad_count;
    end if;
    if holdout_bad_count > 0 then
        raise exception 'raw.content_embedding_holdout contains % rows with non-384 dimensions', holdout_bad_count;
    end if;
end $$;

\echo 'Creating core tables'
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

create table core.demographic (
    postal_code text primary key,
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

create table core.content_embedding (
    product_id bigint primary key,
    embedding vector(384)
);

create table core.fact_sale (
    product_id bigint not null,
    store_id bigint not null,
    soldqty integer not null,
    drawqty integer not null,
    primary key (product_id, store_id),
    check (drawqty >= 0),
    check (soldqty <= drawqty)
);

\echo 'Creating holdout tables'
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

create table holdout.demographic (
    postal_code text primary key,
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

create table holdout.content_embedding (
    product_id bigint primary key,
    embedding vector(384)
);

create table holdout.printing_schedule (
    store_id bigint not null,
    product_id bigint not null,
    primary key (store_id, product_id)
);

\echo 'Promoting raw data into core'
insert into core.dim_product
select *
from raw.dim_product;

insert into core.dim_store
select
    store_id,
    postal_code,
    store_chain,
    region,
    classoftrade,
    case upper(trim(merchandised))
        when 'Y' then true
        when 'N' then false
        else null
    end as merchandised,
    facings,
    pockets
from raw.dim_store;

insert into core.demographic
select
    postal_code,
    max(age_19_and_under) as age_19_and_under,
    max(age_20_to_29) as age_20_to_29,
    max(age_30_to_44) as age_30_to_44,
    max(age_45_to_59) as age_45_to_59,
    max(age_60_and_over) as age_60_and_over,
    max(male) as male,
    max(female) as female,
    max(two_or_more_races) as two_or_more_races,
    max(less_than_10k) as less_than_10k,
    max(between_10k_and_14k) as between_10k_and_14k,
    max(between_15k_and_24k) as between_15k_and_24k,
    max(between_25k_and_34k) as between_25k_and_34k,
    max(between_35k_and_49k) as between_35k_and_49k,
    max(between_50k_and_74k) as between_50k_and_74k,
    max(between_75k_and_99k) as between_75k_and_99k,
    max(between_100k_and_149k) as between_100k_and_149k,
    max(between_150k_and_199k) as between_150k_and_199k,
    max(income_200k_or_more) as income_200k_or_more,
    max(less_than_9th_grade) as less_than_9th_grade,
    max(between_9th_and_12th_grade_no_diploma) as between_9th_and_12th_grade_no_diploma,
    max(high_school_graduate_includes_equivalency) as high_school_graduate_includes_equivalency,
    max(some_college_no_degree) as some_college_no_degree,
    max(associates_degree) as associates_degree,
    max(bachelors_degree) as bachelors_degree,
    max(graduate_or_professional_degree) as graduate_or_professional_degree,
    max(population) as population,
    max(household_type_married_couple_household) as household_type_married_couple_household,
    max(household_type_cohabiting_couple_household) as household_type_cohabiting_couple_household,
    max(household_type_male_householder_no_spouse_partner_present) as household_type_male_householder_no_spouse_partner_present,
    max(household_type_female_householder_no_spouse_partner_present) as household_type_female_householder_no_spouse_partner_present,
    max(family_household) as family_household,
    max(non_family_household) as non_family_household
from raw.demographic
group by postal_code;

insert into core.demographic (
    postal_code,
    age_19_and_under,
    age_20_to_29,
    age_30_to_44,
    age_45_to_59,
    age_60_and_over,
    male,
    female,
    two_or_more_races,
    less_than_10k,
    between_10k_and_14k,
    between_15k_and_24k,
    between_25k_and_34k,
    between_35k_and_49k,
    between_50k_and_74k,
    between_75k_and_99k,
    between_100k_and_149k,
    between_150k_and_199k,
    income_200k_or_more,
    less_than_9th_grade,
    between_9th_and_12th_grade_no_diploma,
    high_school_graduate_includes_equivalency,
    some_college_no_degree,
    associates_degree,
    bachelors_degree,
    graduate_or_professional_degree,
    population,
    household_type_married_couple_household,
    household_type_cohabiting_couple_household,
    household_type_male_householder_no_spouse_partner_present,
    household_type_female_householder_no_spouse_partner_present,
    family_household,
    non_family_household
)
select
    distinct ds.postal_code,
    null::double precision, null::double precision, null::double precision, null::double precision, null::double precision,
    null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision,
    null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision,
    null::double precision,
    null::double precision, null::double precision, null::double precision, null::double precision, null::double precision, null::double precision
from raw.dim_store ds
left join core.demographic d
    on d.postal_code = ds.postal_code
where ds.postal_code is not null
  and d.postal_code is null;

insert into core.content_embedding
select
    ce.product_id,
    core.parse_embedding_text(ce.embedding_text) as embedding
from raw.content_embedding ce
join core.dim_product dp on dp.product_id = ce.product_id;

insert into core.fact_sale
select *
from raw.fact_sale;

\echo 'Promoting raw data into holdout'
insert into holdout.dim_product
select *
from raw.dim_product_holdout;

insert into holdout.dim_store
select
    store_id,
    postal_code,
    store_chain,
    region,
    classoftrade,
    case upper(trim(merchandised))
        when 'Y' then true
        when 'N' then false
        else null
    end as merchandised,
    facings,
    pockets
from raw.dim_store_holdout;

insert into holdout.demographic
select
    postal_code,
    max(age_19_and_under) as age_19_and_under,
    max(age_20_to_29) as age_20_to_29,
    max(age_30_to_44) as age_30_to_44,
    max(age_45_to_59) as age_45_to_59,
    max(age_60_and_over) as age_60_and_over,
    max(male) as male,
    max(female) as female,
    max(two_or_more_races) as two_or_more_races,
    max(less_than_10k) as less_than_10k,
    max(between_10k_and_14k) as between_10k_and_14k,
    max(between_15k_and_24k) as between_15k_and_24k,
    max(between_25k_and_34k) as between_25k_and_34k,
    max(between_35k_and_49k) as between_35k_and_49k,
    max(between_50k_and_74k) as between_50k_and_74k,
    max(between_75k_and_99k) as between_75k_and_99k,
    max(between_100k_and_149k) as between_100k_and_149k,
    max(between_150k_and_199k) as between_150k_and_199k,
    max(income_200k_or_more) as income_200k_or_more,
    max(less_than_9th_grade) as less_than_9th_grade,
    max(between_9th_and_12th_grade_no_diploma) as between_9th_and_12th_grade_no_diploma,
    max(high_school_graduate_includes_equivalency) as high_school_graduate_includes_equivalency,
    max(some_college_no_degree) as some_college_no_degree,
    max(associates_degree) as associates_degree,
    max(bachelors_degree) as bachelors_degree,
    max(graduate_or_professional_degree) as graduate_or_professional_degree,
    max(population) as population,
    max(household_type_married_couple_household) as household_type_married_couple_household,
    max(household_type_cohabiting_couple_household) as household_type_cohabiting_couple_household,
    max(household_type_male_householder_no_spouse_partner_present) as household_type_male_householder_no_spouse_partner_present,
    max(household_type_female_householder_no_spouse_partner_present) as household_type_female_householder_no_spouse_partner_present,
    max(family_household) as family_household,
    max(non_family_household) as non_family_household
from raw.demographic_holdout
group by postal_code;

insert into holdout.content_embedding
select
    ce.product_id,
    core.parse_embedding_text(ce.embedding_text) as embedding
from raw.content_embedding_holdout ce
join holdout.dim_product dp on dp.product_id = ce.product_id;

insert into holdout.printing_schedule
select *
from raw.printing_schedule_holdout;

\echo 'Adding foreign keys and indexes'
alter table core.fact_sale
    add constraint fact_sale_product_fk
    foreign key (product_id) references core.dim_product(product_id);

alter table core.fact_sale
    add constraint fact_sale_store_fk
    foreign key (store_id) references core.dim_store(store_id);

alter table core.content_embedding
    add constraint content_embedding_product_fk
    foreign key (product_id) references core.dim_product(product_id);

alter table core.dim_store
    add constraint dim_store_demographic_fk
    foreign key (postal_code) references core.demographic(postal_code);

alter table holdout.printing_schedule
    add constraint printing_schedule_product_fk
    foreign key (product_id) references holdout.dim_product(product_id);

alter table holdout.printing_schedule
    add constraint printing_schedule_store_fk
    foreign key (store_id) references holdout.dim_store(store_id);

alter table holdout.content_embedding
    add constraint holdout_content_embedding_product_fk
    foreign key (product_id) references holdout.dim_product(product_id);

create index fact_sale_store_id_idx on core.fact_sale (store_id);
create index fact_sale_product_id_idx on core.fact_sale (product_id);
create index core_dim_product_type_idx on core.dim_product (type);
create index core_dim_product_onsaledate_idx on core.dim_product (onsaledate);
create index core_dim_store_postal_code_idx on core.dim_store (postal_code);
create index holdout_printing_schedule_store_id_idx on holdout.printing_schedule (store_id);
create index holdout_printing_schedule_product_id_idx on holdout.printing_schedule (product_id);
create index holdout_dim_product_onsaledate_idx on holdout.dim_product (onsaledate);
create index holdout_dim_store_postal_code_idx on holdout.dim_store (postal_code);

\echo 'Promotion summary'
select 'core.fact_sale.rows' as metric, count(*)::bigint as value from core.fact_sale;
select 'core.dim_product.rows' as metric, count(*)::bigint as value from core.dim_product;
select 'core.dim_store.rows' as metric, count(*)::bigint as value from core.dim_store;
select 'core.demographic.rows' as metric, count(*)::bigint as value from core.demographic;
select 'core.content_embedding.rows' as metric, count(*)::bigint as value from core.content_embedding;
select 'core.dim_product.products_without_embeddings' as metric, count(*)::bigint as value
from core.dim_product dp
left join core.content_embedding ce on ce.product_id = dp.product_id
where ce.product_id is null;
select 'core.content_embedding.orphan_rows_dropped' as metric, count(*)::bigint as value
from raw.content_embedding ce
left join core.dim_product dp on dp.product_id = ce.product_id
where dp.product_id is null;
select 'core.dim_store.postal_code_without_demographic' as metric, count(*)::bigint as value
from core.dim_store ds
left join core.demographic d on d.postal_code = ds.postal_code
where ds.postal_code is not null and d.postal_code is null;
select 'holdout.printing_schedule.rows' as metric, count(*)::bigint as value from holdout.printing_schedule;
select 'holdout.dim_product.rows' as metric, count(*)::bigint as value from holdout.dim_product;
select 'holdout.dim_store.rows' as metric, count(*)::bigint as value from holdout.dim_store;
select 'holdout.demographic.rows' as metric, count(*)::bigint as value from holdout.demographic;
select 'holdout.content_embedding.rows' as metric, count(*)::bigint as value from holdout.content_embedding;
select 'holdout.dim_product.products_without_embeddings' as metric, count(*)::bigint as value
from holdout.dim_product dp
left join holdout.content_embedding ce on ce.product_id = dp.product_id
where ce.product_id is null;
select 'holdout.content_embedding.orphan_rows_dropped' as metric, count(*)::bigint as value
from raw.content_embedding_holdout ce
left join holdout.dim_product dp on dp.product_id = ce.product_id
where dp.product_id is null;
select 'holdout.dim_store.postal_code_without_demographic' as metric, count(*)::bigint as value
from holdout.dim_store ds
left join holdout.demographic d on d.postal_code = ds.postal_code
where ds.postal_code is not null and d.postal_code is null;

\echo 'PaperRush database setup complete'
