import os
import pandas as pd
import support_functions
from tqdm import tqdm
import argparse
import sys
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Schema linking attributes")
    parser.add_argument("--path_to_mimic", type=str, help='')
    args = parser.parse_args(sys.argv[1:])

    my_train_split = json.load(open("statics/ood_train.json", 'r'))
    my_val_split = json.load(open("statics/ood_val.json", 'r'))
    ehrsql_train = my_train_split + my_val_split

    # read original test data
    ehrsql_test = json.load(open("statics/data.json", 'r'))['data']


    db_path = "/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/data/mimic_iv/"
    all_files = os.listdir(db_path)
    csv_files = list(filter(lambda x: x.endswith('.csv'), all_files))
    cols_list, value_list, tables_list = [], [], []
    for file in csv_files:
        df_dict = pd.read_csv(os.path.join(db_path, file)).to_dict(orient="list")
        for key, val_list in df_dict.items():
            value_list += val_list
            cols_list += key.split('_')

        file_name = file.split('.')[0]
        table_parts = file_name.split('_')
        tables_list += table_parts

    value_set = set(value_list)
    cols_set = set(cols_list)
    tables_set = set(tables_list)

    all_vals = value_set.union(cols_set).union(tables_set)
    all_vals_list = list(all_vals)

    all_pr_vals_set = []
    for word in tqdm(all_vals_list):
        pr_value = support_functions.process_word(word)
        all_pr_vals_set.append(pr_value)

    all_pr_vals_set = set(all_pr_vals_set)

    train_id2match_number = dict()
    for sample in tqdm(ehrsql_train):
        id_ = sample['id']
        question = sample['question']
        match_number = len(support_functions.find_match(question, all_pr_vals_set))
        train_id2match_number[id_] = match_number

    test_id2match_number = dict()
    for sample in tqdm(ehrsql_test):
        id_ = sample['id']
        question = sample['question']
        match_number = len(support_functions.find_match(question, all_pr_vals_set))
        test_id2match_number[id_] = match_number

    support_functions.write_json(train_id2match_number, 'statics/full_train_id2match_number.json')
    support_functions.write_json(test_id2match_number, 'statics/test_id2match_number.json')