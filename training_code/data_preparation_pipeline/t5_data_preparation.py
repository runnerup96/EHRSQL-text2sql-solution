import json
import os
import argparse
import sys
import tqdm
import t5_preparation_utils
import utils.data_processing_utils as processing_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser('The testing components of')
    parser.add_argument('--file_path', type=str, help='')
    parser.add_argument('--split_name', type=str, help='')
    args = parser.parse_args(sys.argv[1:])

    assert args.split_name in ['train', 'eval', 'test', 'full_text2sql_train', 'full_ehrsql_train']

    dataset_table_info = os.path.join(os.environ["EHRSQL_STATIC"], "data/mimic_iv/tables.json")
    mimic_tables_json = json.load(open(dataset_table_info, 'r'))

    db_id_to_schema_string = dict()
    for table_json in mimic_tables_json:
        db_id = table_json["db_id"]
        db_id_to_schema_string[db_id] = t5_preparation_utils.get_schema_string(table_json)

    train_samples = json.load(open(args.file_path, 'r'))
    if "data" in train_samples:
        train_samples = train_samples["data"]

    samples_list = []
    db_id = 'mimic_iv'
    schema_str = db_id_to_schema_string[db_id]
    for sample in tqdm.tqdm(train_samples):
        nl = processing_utils.process_input_question(sample['question'])
        source = processing_utils.prepare_model_input(db_id=db_id, question=nl,
                                                      schema_str=schema_str)
        id_ = sample['id']

        if args.split_name in ['train', 'eval', "full_text2sql_train"]:
            sql = processing_utils.normalize_sql_query(sample['query'])
            target = processing_utils.prepare_model_target(db_id=db_id, query=sql)
            samples_list.append((id_, source, target))
        elif args.split_name in ['test', "full_ehrsql_train"]:
            samples_list.append((id_, source))

    save_path = os.path.join(os.environ["EHRSQL_STATIC"], 'data/tmp_train_data')

    save_file_path = os.path.join(save_path, f'{args.split_name}_for_t5.tsv')
    t5_preparation_utils.write_tsv(samples_list, save_file_path, len(samples_list[0]))

    print(f'File for split {args.split_name} saved to {save_file_path}')