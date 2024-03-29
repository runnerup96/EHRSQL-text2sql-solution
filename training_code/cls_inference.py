import argparse
import os
import pandas as pd
import torch
import sys
import training_utils
from tqdm import tqdm
import evaluation
import cls_dataset
from question_classifier import QuestionClassifier
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser('The testing components of')

    parser.add_argument('--original_model_path', default="bert", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--test_file', required=True, type=str, help='path to the testing file')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cls_treshold', default=0.5, type=float, help="Classifier treshold")
    parser.add_argument('--output_path', required=True, type=str, help='path to the trained model')
    parser.add_argument('--try_one_batch', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')

    args = parser.parse_args(sys.argv[1:])

    set_seed(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path, use_fast=True)

    test_samples = training_utils.read_cls_dataset(dataset_path=args.test_file,
                                                   split_name="test",
                                                   tokenizer=tokenizer,
                                                   input_max_length=args.max_seq_len)

    if args.try_one_batch:
        test_samples = test_samples[-args.batch_size:]
        test_dataset = cls_dataset.CLSDataset(test_samples, tokenizer, args.max_seq_len, args.device)
    else:
        test_dataset = cls_dataset.CLSDataset(test_samples, tokenizer, args.max_seq_len, args.device)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    cls_classifier = QuestionClassifier(pretrained_model_name=args.original_model_path).to(args.device)
    trained_model_path = os.path.join(args.output_path, "question_classifier.bin")
    cls_classifier.load_state_dict(torch.load(trained_model_path))

    curr_step=1
    total_pbar = tqdm(total=len(test_dataloader), desc='Total train iters')
    cls_classifier.eval()
    proba_list, pred_list= [], []
    for batch in test_dataloader:
        input, attention_mask, target = batch["input_ids"], batch["attention_mask"], batch["target"]
        if target == []:
            target = None
        proba_batch, _ = cls_classifier.forward(input_ids=input, attention_mask=attention_mask, target=target)
        proba_list += [proba.item() for proba in proba_batch]
        if args.cls_treshold:
            preds = (proba_batch >= args.cls_treshold).float()
            pred_list += [pred.item() for pred in preds]

        torch.cuda.empty_cache()
        curr_step += 1
        total_pbar.update(1)

    result = {
        "id": [sample["id"] for sample in test_samples],
        "question": [sample["question"] for sample in test_samples],
        "target": [sample["target"] for sample in test_samples],
        "predicted_proba": proba_list,
        f"predicted_class@{args.cls_treshold}": pred_list
    }

    if args.do_eval:
        metrics = evaluation.calculate_specificity_sensitivity(torch.tensor(proba_list),
                                                               torch.tensor(result['target']), args.cls_treshold)
        print(f'\nResult at treshold={args.cls_treshold}')
        print(metrics)
        print('Avg score: ', (metrics['specificity'] + metrics['sensitivity']) / 2)

    test_file_name = os.path.basename(args.test_file).split(".")[0]
    df = pd.DataFrame(result)
    save_path = os.path.join(args.output_path, f"{test_file_name}_prediction.tsv")
    df.to_csv(save_path, sep='\t', index=False)
    print(f"File {args.test_file} was precessed and saved to {save_path}")



