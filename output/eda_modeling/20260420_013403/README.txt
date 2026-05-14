EDA modeling diagnostics output directory: /Users/norbert.jaworski/Documents/uni/2 rok/understandingBusiness/project/output/eda_modeling/20260420_013403

Main diagnostics included:
- integrity_overview.csv
- join_coverage.csv
- soldqty_distribution_by_type.csv
- operating_slice_type.csv
- operating_slice_segment.csv
- operating_slice_subsegment.csv
- operating_slice_chain.csv
- operating_slice_class.csv
- operating_slice_region.csv
- store_performance_panel.csv
- store_dispersion_by_chain.csv
- merchandising_effect.csv
- weeklies_issue_panel.csv
- weeklies_store_title_recurrence.csv
- weeklies_title_persistence.csv
- weeklies_store_attribute_sensitivity.csv
- sip_segment_similarity.csv
- sip_embedding_neighbor_diagnostic.csv
- sip_explanatory_power.csv
- holdout_modeling_mix.csv

Key target note:
- Recommended primary regression target is pmax(SoldQty, 0), not raw SoldQty.

Key integrity facts:
- core.fact_sale rows: 16590626
- negative sales rows: 11719
- stockout proxy rows: 403551
- average sell-through: 0.1202
- average oversupply units: 4.3165

Join coverage checks:
- holdout_product_to_embedding: missing_rows=  1 / total_rows=     298
- train_product_to_embedding: missing_rows= 37 / total_rows=    5324
- holdout_store_to_demographic: missing_rows=311 / total_rows=   27714
- train_store_to_demographic: missing_rows=  0 / total_rows=   29622
- holdout_schedule_to_product: missing_rows=  0 / total_rows=  761531
- holdout_schedule_to_store: missing_rows=  0 / total_rows=  761531
- train_fact_to_product: missing_rows=  0 / total_rows=16590626
- train_fact_to_store: missing_rows=  0 / total_rows=16590626
