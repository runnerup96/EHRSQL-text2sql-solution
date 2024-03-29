import evaluate
from transformers import EvalPrediction
import torch

class Evaluator:
    def __init__(self):
        self.exact_match = evaluate.load('exact_match')

    def evaluate(self, p: EvalPrediction):
        metrics_dict = dict()
        exact_match_metric = self.exact_match.compute(predictions=p.predictions, references=p.label_ids,
                                                      ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')
        metrics_dict.update(exact_match_metric)
        return metrics_dict


def calculate_specificity_sensitivity(probability, target, threshold):
    # Convert probability to binary predictions based on threshold
    predictions = (probability >= threshold).float()
    true_positives = torch.sum((predictions == 1) & (target == 1)).item()

    false_positives = torch.sum((predictions == 1) & (target == 0)).item()

    true_negatives = torch.sum((predictions == 0) & (target == 0)).item()

    false_negatives = torch.sum((predictions == 0) & (target == 1)).item()

    sensitivity = 0
    if true_positives + false_negatives != 0:
        sensitivity = true_positives / (true_positives + false_negatives)

    specificity = 0
    if true_negatives + false_positives != 0:
        specificity = true_negatives / (true_negatives + false_positives)

    return {"specificity": specificity, "sensitivity": sensitivity}