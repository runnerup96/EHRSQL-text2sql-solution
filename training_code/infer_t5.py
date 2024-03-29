import logging
import os
import pickle
from tqdm import tqdm
import torch
from transformers import HfArgumentParser, T5ForConditionalGeneration, AutoTokenizer, AutoConfig, Adafactor, \
    set_seed, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
import hf_arguments
import text2sql_dataset
import training_utils


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        use_fast=True
    )
    tokenizer.add_tokens([" <", " <=", "<>"])

    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                       config=config,
                                                       torch_dtype=torch.bfloat16).to(device)

    test_samples = training_utils.read_t5_tsv_dataset(data_args.test_file,
                                                      tokenizer=tokenizer,
                                                      input_max_length=data_args.max_seq_length,
                                                      output_max_length=data_args.max_output_length)

    if data_args.try_one_batch:
        one_batch_size = 32
        one_batch_samples = test_samples[-one_batch_size:]
        test_dataset = text2sql_dataset.T5FinetuneDataset(one_batch_samples, tokenizer)
    else:
        test_dataset = text2sql_dataset.T5FinetuneDataset(test_samples, tokenizer)

    eval_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=training_args.eval_batch_size)
    ids_list = []
    prediction_list = []
    scores_list = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        ids = batch['id']
        source_ids = batch["input_ids"].to(device)
        outputs = model.generate(input_ids=source_ids, max_length=data_args.max_seq_length,
                                 num_beams=data_args.num_beams,
                                 output_scores=True, return_dict_in_generate=True)

        generated_sequences = outputs["sequences"].cpu() if "cuda" in device else outputs["sequences"]

        entropy_scores = training_utils.maximum_entropy_confidence_score_method(generation_scores=outputs["scores"],
                                                                                device=device)
        entropy_scores = training_utils.truncate_scores(generated_sequences=generated_sequences,
                                                        scores=entropy_scores,
                                                        tokenizer=tokenizer)
        max_entropy_scores = [max(score_list) for score_list in entropy_scores]
        scores_list += max_entropy_scores

        decoded_preds = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        ids_list += ids
        predictions = [training_utils.generated_query_simple_processor(pred) for pred in decoded_preds]
        prediction_list += predictions

    result_dict = dict()
    for id_, pred_sql, score in zip(ids_list, prediction_list, scores_list):
        result_dict[id_] = {
            "sql": pred_sql,
            "score": score
        }

    output_dir = training_args.output_dir
    filename = os.path.basename(data_args.test_file).split('.')[0]
    if data_args.try_one_batch:
        filename = f"{filename}_one_batch_inference_result.pkl"
    else:
        filename = f"{filename}_inference_result.pkl"
    save_path = os.path.join(output_dir, filename)
    logger.info("Writing model predictions to json file...")

    pickle.dump(result_dict, open(save_path, 'wb'))