import os
import json
import sys
import argparse
import pickle
import numpy as np
import postprocessing
import reliability_score
from utils import data_io


def get_threshold_for_maxent(id2maxent, score_dict):
    """
    Determine the optimal threshold for filtering based on maximum entropy and scores.
    """
    values = []
    scores = []
    for key, val in id2maxent.items():
        values.append(val)
        scores.append(score_dict[key])

    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_scores = np.array(scores)[sorted_indices]

    max_score, threshold = 0, -1
    for idx in range(len(sorted_scores)):
        cum_score = sum(sorted_scores[:idx+1])
        if cum_score > max_score:
            # print('cum_score > max_score')
            max_score, threshold = cum_score, sorted_values[idx-1]
    threshold = round(threshold, 3)
    return threshold



if __name__ == "__main__":

    cmd = argparse.ArgumentParser()

    cmd.add_argument('--gold_file_path', required=False, type=str)
    cmd.add_argument('--prediction_file_path',  required=True, type=str)
    cmd.add_argument('--db_path', required=False, type=str)
    cmd.add_argument('--submission_name', required=True, type=str)
    cmd.add_argument('--threshold_value', required=False, type=str)
    cmd.add_argument('--calculate_scores', default=False, action='store_true')


    args = cmd.parse_args(sys.argv[1:])

    my_pred_dict = pickle.load(open(args.prediction_file_path, 'rb'))

    pred_dict_for_eval = dict()
    id2maxent = dict()
    for id_ in my_pred_dict:
        pred_sql, score = my_pred_dict[id_]['sql'], my_pred_dict[id_]['score']
        pr_pred_sql = postprocessing.post_process_sql(pred_sql)
        pred_dict_for_eval[id_] = pr_pred_sql
        id2maxent[id_] = score

    if args.calculate_scores:
        with open(args.gold_file_path) as f:
            gold_dict = json.load(f)

        gold_label_dict = dict()
        for sample in gold_dict:
            gold_label_dict[sample['id']] = sample['query']
        assert set(pred_dict_for_eval) == set(gold_label_dict)

        gold_dict_for_eval = dict()
        for id_ in gold_label_dict:
            gold_sql = gold_label_dict[id_]
            pr_gold_sql = postprocessing.post_process_sql(gold_sql)
            gold_dict_for_eval[id_] = pr_gold_sql

        scores, score_dict = reliability_score.calculate_score(gold_dict_for_eval, pred_dict_for_eval,
                                                               db_path=args.db_path, return_dict=True)
        optimal_treshold = get_threshold_for_maxent(id2maxent, score_dict)
        print('Found treshold: ', optimal_treshold)

    elif args.threshold_value:
        optimal_treshold = float(args.threshold_value)
    else:
        raise "No treshold or gold file for tresh calculation is provided!"

    for id_ in pred_dict_for_eval:
        pred_score = id2maxent[id_]
        if optimal_treshold < pred_score:
            pred_dict_for_eval[id_] = 'null'

    if args.calculate_scores:
        scores = reliability_score.calculate_score(gold_dict_for_eval, pred_dict_for_eval, db_path=args.db_path)
        accuracy0 = reliability_score.penalize(scores, penalty=0)
        accuracy5 = reliability_score.penalize(scores, penalty=5)
        accuracy10 = reliability_score.penalize(scores, penalty=10)
        accuracyN = reliability_score.penalize(scores, penalty=len(scores))

        print('Scores:')
        scores_dict = {
            'accuracy0': accuracy0 * 100,
            'accuracy5': accuracy5 * 100,
            'accuracy10': accuracy10 * 100,
            'accuracyN': accuracyN * 100
        }
        print(scores_dict)

    RESULTS_DIR = "/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/submissions"
    submission_path = os.path.join(RESULTS_DIR, f"{args.submission_name}.json")
    data_io.write_json(submission_path, pred_dict_for_eval)
    result_file_path = os.path.dirname(args.prediction_file_path)
    print(f'Evaluation done! Submission file save to {submission_path}!')
    #https://www.codabench.org/competitions/1889/




