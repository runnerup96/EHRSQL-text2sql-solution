python3 preprocess_db.py \
--data_dir /Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/data/mimic_iv/mimic-iv-clinical-database-demo-2.2 \
--db_name mimic_iv \
--num_patient 100 \
--timeshift \
--current_time "2100-12-31 23:59:00" \
--start_year 2100 \
--time_span 0 \
--out_dir /Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/data/ \
--cur_patient_ratio 0.1