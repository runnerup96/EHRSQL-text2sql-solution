import json
import sys
import argparse
from scoring_program_previous.reliability_score import calculate_score, penalize
from postprocessing import post_process_sql



if __name__ == "__main__":

    cmd = argparse.ArgumentParser()

    cmd.add_argument('--gold_file_path', required=False, type=str)
    cmd.add_argument('--prediction_file_path', required=True, type=str)
    cmd.add_argument('--db_path', required=False, type=str)

    args = cmd.parse_args(sys.argv[1:])


    print('Load Data')
    with open(args.gold_file_path) as f:
        truth = json.load(f)
    with open(args.prediction_file_path) as f:
        prediction = json.load(f)

    print('Checking Accuracy')
    try:
        real_dict = {id_: post_process_sql(truth[id_]) for id_ in truth}
    except TypeError:
        real_dict = {sample['id']: post_process_sql(sample['query']) for sample in truth}
    pred_dict = {id_: post_process_sql(prediction[id_]) for id_ in prediction}
    assert set(real_dict) == set(pred_dict), "IDs do not match"

    scores, result_dict = calculate_score(real_dict, pred_dict, args.db_path, return_dict=True)
    accuracy0 = penalize(scores, penalty=0)
    accuracy5 = penalize(scores, penalty=5)
    accuracy10 = penalize(scores, penalty=10)
    accuracyN = penalize(scores, penalty=len(scores))

    print('Scores:')
    scores_dict = {
        'accuracy0': accuracy0*100,
        'accuracy5': accuracy5*100,
        'accuracy10': accuracy10*100,
        'accuracyN': accuracyN*100
    }
    print(scores_dict)

    json.dump(result_dict, open("/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/submissions/my_dev_scores.json", 'w'),
              ensure_ascii=False, indent=4)
