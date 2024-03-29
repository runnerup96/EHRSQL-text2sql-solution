import argparse
from sklearn.preprocessing import StandardScaler
import support_functions
import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import sys



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Final system attributes")

    parser.add_argument("--db_path", type=str, help='')
    args = parser.parse_args(sys.argv[1:])

    # read original training data
    # we splited data into two splits for our experiements
    my_train_split = json.load(open("statics/ood_train.json", 'r'))
    my_val_split = json.load(open("statics/ood_val.json", 'r'))
    ehrsql_train = my_train_split + my_val_split

    # read original test data
    ehrsql_test = json.load(open("statics/data.json", 'r'))['data']

    # read T5-3b prediction
    t5_ehrsql_test_prediction = pickle.load(open("statics/ehrsql_test_for_t5_inference_result.pkl", 'rb'))

    # schema and values intersection score with original question
    ehrsql_train_match_features = json.load(open("statics/full_train_id2match_number.json", 'r'))
    ehrsq_test_match_features = json.load(open("statics/test_id2match_number.json", 'r'))


    # calculate question length feature for train and test
    ehrsql_train_qs_length_feature = dict()
    for sample in tqdm(ehrsql_train):
        id_ = sample['id']
        question = sample['question']
        ehrsql_train_qs_length_feature[id_] = support_functions.get_processed_question_length(question)

    ehrsql_test_qs_length_feature = dict()
    for sample in tqdm(ehrsql_test):
        id_ = sample['id']
        question = sample['question']
        ehrsql_test_qs_length_feature[id_] = support_functions.get_processed_question_length(question)

    # calculate question starting words feature for train and test
    ehrsql_train_start_word_feature = dict()
    for sample in tqdm(ehrsql_train):
        id_ = sample['id']
        question = sample['question']
        ehrsql_train_start_word_feature[id_] = support_functions.get_starting_words_status(question)

    ehrsql_test_start_word_feature = dict()
    for sample in tqdm(ehrsql_test):
        id_ = sample['id']
        question = sample['question']
        ehrsql_test_start_word_feature[id_] = support_functions.get_starting_words_status(question)

    # merge all 3 features into one training dict + target
    all_features_train = dict()
    for sample in ehrsql_train:
        id_ = sample['id']

        intersect_feature = ehrsql_train_match_features[id_]
        qs_length_feature = ehrsql_train_qs_length_feature[id_]
        start_word_feature = 1 if ehrsql_train_start_word_feature[id_] else 0

        target = 0 if sample['query'] == 'null' else 1

        all_features_train[id_] = {
            "intersect_feature": intersect_feature,
            "qs_length_feature": qs_length_feature,
            "start_word_feature": start_word_feature,
            "target": target
        }

    all_train_df = pd.DataFrame().from_dict(all_features_train, orient='index')

    # merge all 3 features into one testing dict
    all_features_test = dict()
    for sample in ehrsql_test:
        id_ = sample['id']

        intersect_feature = ehrsq_test_match_features[id_]
        qs_length_feature = ehrsql_test_qs_length_feature[id_]
        start_word_feature = 1 if ehrsql_test_start_word_feature[id_] == True else 0

        all_features_test[id_] = {
            "intersect_feature": intersect_feature,
            "qs_length_feature": qs_length_feature,
            "start_word_feature": start_word_feature,
        }

    all_test_df = pd.DataFrame().from_dict(all_features_test, orient='index')

    # scale result dataframes for training model
    my_features = ["intersect_feature", "qs_length_feature", "start_word_feature"]
    target = "target"

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_train_df[my_features].values)
    scaled_all_train_df = pd.DataFrame(scaled_features, columns=my_features, index=all_train_df.index)
    scaled_all_train_df[target] = all_train_df[target]
    scaled_all_train_df = scaled_all_train_df.sample(n=len(all_train_df), random_state=42)

    scaled_all_test_df = pd.DataFrame(scaler.transform(all_test_df[my_features].values), columns=my_features,
                                      index=all_test_df.index)

    # training logistic regression to identify null questions
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(scaled_all_train_df[my_features], scaled_all_train_df[target])

    preds, preds_proba = log_reg.predict(scaled_all_test_df[my_features]), log_reg.predict_proba(
        scaled_all_test_df[my_features])
    null_score_confidence = preds_proba[:, 0].reshape(-1, 1)

    # scale scores to smooth log reg probability
    scaler = MinMaxScaler()
    null_score_confidence = scaler.fit_transform(null_score_confidence)

    # we take top-20% percent of null score confident samples
    null_known_ratio = 0.1995
    null_totals = int(len(scaled_all_test_df) * null_known_ratio)
    test_index2score = {idx: score for idx, score in zip(list(scaled_all_test_df.index), null_score_confidence)}
    sorted_test_index2score = sorted(test_index2score.items(), key=lambda x: x[1], reverse=True)
    null_samples = sorted_test_index2score[:null_totals]
    sql_samples = sorted_test_index2score[null_totals:]

    meta_model_submission = dict()
    for sample in null_samples:
        id_, _ = sample
        meta_model_submission[id_] = 'null'

    for sample in sql_samples:
        id_, _ = sample
        meta_model_submission[id_] = 'not null'

    id2pred = dict()

    # run system prediction with make_prediction_with_meta_model
    for id_ in tqdm(t5_ehrsql_test_prediction):
        result = support_functions.make_prediction_with_meta_model(id_, t5_preds=t5_ehrsql_test_prediction,
                                                                   meta_model_preds=meta_model_submission,
                                                                   db_path=args.db_path)
        id2pred[id_] = result

    # assert the result with our top performing solution to make sure we got the same result as in leaderboard
    my_top_submission = json.load(open("statics/meta_model_v6_submission_meta_model_final.json", 'r'))

    assert id2pred == my_top_submission

    print('Prepared final result solution!')