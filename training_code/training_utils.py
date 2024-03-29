import json
import torch
import pandas as pd
from transformers import EvalPrediction
from transformers.trainer_utils import EvalLoopOutput
import data_processing_utils as processing_utils


def generated_query_simple_processor(query):
    query = query.split('|')[-1]
    query = query.strip()
    query = query.replace('< -', '<-')
    return query


def original_query_simple_processor(query):
    query = query.split('|')[-1]
    query = query.strip()
    return query


def model_post_processing_function(examples: list, outputs: EvalLoopOutput, tokenizer):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    predictions = [generated_query_simple_processor(pred) for pred in decoded_preds]
    raw_references = [original_query_simple_processor(sample['target']) for sample in examples]

    return EvalPrediction(predictions=predictions, label_ids=raw_references)


def read_t5_tsv_dataset(dataset_path, tokenizer, input_max_length, output_max_length):
    samples = []
    df = pd.read_csv(dataset_path, sep='\t', header=None, keep_default_na=False, na_values=['NaN'])

    if df.shape[1] == 3:
        df.columns = ['id', 'source', 'target']
    elif df.shape[1] == 2:
        df.columns = ['id', 'source']
        df['target'] = None

    for id_, source, target in zip(df['id'].to_list(), df['source'].tolist(), df['target'].tolist()):
        formatted_source = 'semanticparse: ' + source
        source_tokens = tokenizer.encode(formatted_source, add_special_tokens=False, truncation=True,
                                         max_length=input_max_length)

        if target:
            formatted_target = target
            target_tokens = tokenizer.encode(formatted_target, add_special_tokens=False, truncation=True,
                                             max_length=output_max_length)
        else:
            target_tokens = None

        samples.append({'id': id_,
                        'source_tokens': source_tokens,
                        'target_tokens': target_tokens,
                        'source': source,
                        'target': target})
    return samples

def read_pair_based_dataset(dataset_path, split_name, tokenizer, input_max_length, upsample_ratio=3):
    dataset = json.load(open(dataset_path, 'r'))

    samples = []
    for sample in dataset:
        id_ = sample['id']
        question = sample['question']
        query = sample['query']
        target = sample['target']

        process_question = processing_utils.process_input_question(question)
        process_query = query.strip()
        result_input = f"{process_question} {tokenizer.sep_token} {process_query}"
        input_tokens = tokenizer.encode(result_input, add_special_tokens=False, truncation=True,
                                        max_length=input_max_length)

        if target == 0 and split_name == 'train':
            for _ in range(upsample_ratio+1):
                samples.append({
                    'id': id_,
                    "question": question,
                    'input_tokens': input_tokens,
                    'target': target,
                })
        else:
            samples.append({
                'id': id_,
                "question": question,
                'input_tokens': input_tokens,
                'target': target,
            })
    return samples


def read_cls_dataset(dataset_path, split_name, tokenizer, input_max_length, upsample_ratio=3):
    dataset = json.load(open(dataset_path, 'r'))
    if "data" in dataset:
        dataset = dataset["data"]
    samples = []
    for sample in dataset:
        id_ = sample['id']
        question = sample['question']
        process_input = processing_utils.process_input_question(question)
        input_tokens = tokenizer.encode(process_input, add_special_tokens=False, truncation=True,
                                        max_length=input_max_length)

        query = sample.get('query')
        if query:
            target = 0 if query == 'null' else 1
        else:
            target = None
        if target == 0 and split_name == 'train':
            for _ in range(upsample_ratio+1):
                samples.append({
                    'id': id_,
                    "question": question,
                    'input_tokens': input_tokens,
                    'target': target,
                })
        else:
            samples.append({
                'id': id_,
                "question": question,
                'input_tokens': input_tokens,
                'target': target,
            })
    return samples


def maximum_entropy_confidence_score_method(generation_scores, device):
    # TODO: Work with beans to samples ratio here
    logits = torch.stack(generation_scores, dim=1)[:: 1]
    logits = logits.cpu() if "cuda" in device else logits
    probs = torch.softmax(logits, dim=2).float()
    log_probs = torch.log_softmax(logits, dim=2).float()
    entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()

    return entropies

def truncate_scores(generated_sequences, scores, tokenizer):
    scores_list = []
    for idx in range(len(generated_sequences)):
        pred_tensor = generated_sequences[idx][1:]
        scores_truncated = scores[idx].tolist()

        # Truncate the prediction at the end-of-sequence token, if present.
        if tokenizer.eos_token_id in pred_tensor:
            pred_eos_idx = torch.nonzero(pred_tensor == tokenizer.eos_token_id)[0].item()
            scores_truncated = scores_truncated[: pred_eos_idx + 1]

        scores_list.append(scores_truncated)

    return scores_list
