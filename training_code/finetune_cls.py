import argparse
import os
import math
import torch
import sys
import training_utils
from tqdm import tqdm
import evaluation
import cls_dataset
from question_classifier import QuestionClassifier
from transformers import AutoTokenizer, AutoConfig, \
    set_seed, get_linear_schedule_with_warmup, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser('The testing components of')


    parser.add_argument('--training_type', default="only_question", type=str, help='')
    parser.add_argument('--model_path', default="cuda", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_file_path', required=True, type=str, help='path to the training corpus')
    parser.add_argument('--eval_file_path', required=True, type=str, help='path to the eval corpus')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--upsample_ratio', default=3, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=25, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--output_path', required=True, type=str, help='path to the training corpus')
    parser.add_argument('--try_one_batch', default=False, action='store_true')

    args = parser.parse_args(sys.argv[1:])

    set_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    tensorboard_layout = {
        "Question Classifier": {
            "loss": ["Multiline", ["loss/train", "loss/val"]],
            "accuracy": ["Multiline", ["ba_score/val", "specificity/val", "sensitivity/val"]],
            "params": ["Multiline", ["learning_rate/train"]]
        },
    }
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.output_path, "logs"))
    tensorboard_writer.add_custom_scalars(tensorboard_layout)


    config = AutoConfig.from_pretrained(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    train_file_path = args.train_file_path
    eval_file_path = args.eval_file_path
    if args.training_type == 'only_question':
        train_samples = training_utils.read_cls_dataset(dataset_path=train_file_path,
                                                        split_name="train",
                                                        tokenizer=tokenizer,
                                                        input_max_length=args.max_seq_len,
                                                        upsample_ratio=args.upsample_ratio)

        test_samples = training_utils.read_cls_dataset(dataset_path=eval_file_path,
                                                               split_name="test",
                                                               tokenizer=tokenizer,
                                                               input_max_length=args.max_seq_len,
                                                               upsample_ratio=args.upsample_ratio)
    elif args.training_type == 'question_and_query':
        train_samples = training_utils.read_pair_based_dataset(dataset_path=train_file_path,
                                                        split_name="train",
                                                        tokenizer=tokenizer,
                                                        input_max_length=args.max_seq_len,
                                                        upsample_ratio=args.upsample_ratio)

        test_samples = training_utils.read_pair_based_dataset(dataset_path=eval_file_path,
                                                       split_name="test",
                                                       tokenizer=tokenizer,
                                                       input_max_length=args.max_seq_len,
                                                       upsample_ratio=args.upsample_ratio)


    if args.try_one_batch:
        one_batch_samples = train_samples[-args.batch_size:]
        test_samples = train_samples[-args.batch_size:]
        train_dataset = cls_dataset.CLSDataset(one_batch_samples, tokenizer, args.max_seq_len, args.device)
        test_dataset = cls_dataset.CLSDataset(one_batch_samples, tokenizer, args.max_seq_len, args.device)
    else:
        train_dataset = cls_dataset.CLSDataset(train_samples, tokenizer, args.max_seq_len, args.device)
        test_dataset = cls_dataset.CLSDataset(test_samples, tokenizer, args.max_seq_len, args.device)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    cls_classifier = QuestionClassifier(pretrained_model_name=args.model_path).to(args.device)

    optimizer = AdamW(cls_classifier.parameters(), lr=args.lr)

    batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.floor(len(train_dataset) / batch_size)
    total_train_steps = args.epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * total_train_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=total_train_steps)

    curr_step = 1
    best_balanced_accuracy = 0.5
    total_pbar = tqdm(total=total_train_steps, desc='Total train iters')
    model_name = os.path.join(args.output_path, f"question_classifier.bin")
    for epoch in range(args.epochs):
        cls_classifier.train()
        for batch in train_dataloader:
            input, attention_mask, target = batch["input_ids"], batch["attention_mask"], batch["target"]

            _, loss = cls_classifier.forward(input_ids=input, attention_mask=attention_mask, target=target)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if curr_step % args.gradient_accumulation_steps == 0:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if curr_step % args.log_steps == 0:
                tensorboard_writer.add_scalar("loss/train", loss.item(), curr_step)

            tensorboard_writer.add_scalar("learning_rate/train", lr_scheduler.get_last_lr()[0], curr_step)

            if curr_step % args.valid_steps == 0:
                cls_classifier.eval()
                eval_metrics = {"specificity": 0, "sensitivity": 0}
                for batch in test_dataloader:
                    input, attention_mask, target = batch["input_ids"], batch["attention_mask"], batch["target"]

                    proba, loss = cls_classifier.forward(input, attention_mask, target)

                    metrics = evaluation.calculate_specificity_sensitivity(proba, target, 0.5)
                    for key in metrics:
                        eval_metrics[key] += metrics[key]

                for key in eval_metrics:
                    eval_metrics[key] /= len(test_dataloader)
                balanced_accuracy = (eval_metrics['specificity'] + eval_metrics['sensitivity']) / 2

                tensorboard_writer.add_scalar("specificity/val", eval_metrics["specificity"], curr_step)
                tensorboard_writer.add_scalar("sensitivity/val", eval_metrics['sensitivity'], curr_step)
                tensorboard_writer.add_scalar("ba_score/val", balanced_accuracy, curr_step)

                tensorboard_writer.add_scalar("loss/val", loss.item(), curr_step)


                print(f"specificity: {round(eval_metrics['specificity'], 3)}, "
                      f"sensitivity:{round(eval_metrics['sensitivity'], 3)}, "
                      f"ba_score:{round(balanced_accuracy, 3)}")

                if balanced_accuracy > best_balanced_accuracy:
                    print("New best BA: ", round(balanced_accuracy, 3))
                    best_balanced_accuracy = balanced_accuracy

                    torch.save(cls_classifier.state_dict(), model_name)

            curr_step += 1
            total_pbar.update(1)

    print(f'Training completed! Best model here {model_name}')
