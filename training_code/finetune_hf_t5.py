import logging
import os
import math

import torch
from transformers import HfArgumentParser, T5ForConditionalGeneration, AutoTokenizer, AutoConfig, Adafactor, \
    set_seed, Seq2SeqTrainingArguments, get_cosine_schedule_with_warmup
from transformers.trainer_utils import get_last_checkpoint
import evaluation
import hf_arguments
import text2sql_dataset
import training_utils
from sp_seq2seq_trainer import SemanticParsingSeq2SeqTrainer

def main():
    logger = logging.getLogger(__name__)

    hf_parser = HfArgumentParser((hf_arguments.ModelArguments, hf_arguments.DataTrainingArguments,
                                  Seq2SeqTrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    tokenizer.add_tokens([" <", " <=", "<>"])


    train_samples = training_utils.read_t5_tsv_dataset(data_args.train_file,
                                                       tokenizer=tokenizer,
                                                       input_max_length=data_args.max_seq_length,
                                                       output_max_length=data_args.max_output_length)
    test_samples = training_utils.read_t5_tsv_dataset(data_args.validation_file,
                                                      tokenizer=tokenizer,
                                                      input_max_length=data_args.max_seq_length,
                                                      output_max_length=data_args.max_output_length)

    if data_args.try_one_batch:
        one_batch_size = 32
        one_batch_samples = train_samples[-one_batch_size:]
        test_samples = train_samples[-one_batch_size:]
        train_dataset = text2sql_dataset.T5FinetuneDataset(one_batch_samples, tokenizer)
        test_dataset = text2sql_dataset.T5FinetuneDataset(one_batch_samples, tokenizer)
    else:
        train_dataset = text2sql_dataset.T5FinetuneDataset(train_samples, tokenizer)
        test_dataset = text2sql_dataset.T5FinetuneDataset(test_samples, tokenizer)


    # Prepare model & optimizer
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                       config=config,
                                                       torch_dtype=torch.bfloat16)

    optimizer = Adafactor(model.parameters(), lr=training_args.learning_rate,
                          scale_parameter=False, relative_step=False, clip_threshold=1.0,
                          warmup_init=False)
    # train step calculation
    # https://stackoverflow.com/questions/71607906/understanding-gpu-usage-huggingface-classification-total-optimization-steps

    batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = math.floor(len(train_dataset) / batch_size)
    total_train_steps = training_args.num_train_epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * total_train_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=total_train_steps)
    print('My total train steps: ', total_train_steps)

    # prepare evaluation class
    evaluator = evaluation.Evaluator()

    trainer = SemanticParsingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        eval_examples=test_samples if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=evaluator.evaluate,
        optimizers=(optimizer, lr_scheduler),
        post_process_function=training_utils.model_post_processing_function,
    )

    if training_args.do_train:
        last_checkpoint = None
        if os.path.isdir(
                training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=data_args.max_seq_length,
                                   num_beams=data_args.num_beams, metric_key_prefix="eval",
                                   output_save_dir=training_args.output_dir)

        if 'predictions' in metrics:
            output_dir = training_args.output_dir
            filename = os.path.basename(output_dir).split('.')[0]
            filename = f"{filename}_prediction.txt"
            save_path = os.path.join(output_dir, filename)
            with open(save_path, 'w') as f:
                for pred in metrics['predictions']:
                    f.write(f"{pred} \n")
            metrics.pop('predictions')

        metrics["eval_samples"] = len(test_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


        # res = trainer.predict(test_dataset, tokenizer=tokenizer)
        # # Save the prediction files for spider evaluation
        # prediction_list = []
        # for pred_idx, pred_id in enumerate(res.predictions):
        #     prediction_list.append(pred_id)
        #
        # output_dir = training_args.output_dir
        # filename = os.path.basename(output_dir).split('.')[0]
        # filename = f"{filename}_prediction.txt"
        # save_path = os.path.join(output_dir, filename)
        #
        # logger.info("Writing model predictions to txt file...")
        # with open(save_path, 'w') as f:
        #     for line in prediction_list:
        #         f.write(f"{line}\n")


if __name__ == "__main__":
    main()
