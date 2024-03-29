# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Semantic Parsing tasks
"""
import os.path
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_torch_tpu_available():
    pass


class SemanticParsingSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples=None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
            output_save_dir: Optional[str]=None
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
            # pdb.set_trace()
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None and self.tokenizer is not None:
            eval_preds = self.post_process_function(eval_examples, output, self.tokenizer)
            metrics = self.compute_metrics(eval_preds)
            preds_list = eval_preds.predictions
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            metrics.update(output.metrics)

            in_progress_preds_file = os.path.join(self.args.output_dir, 'in_progress_preds.txt')
            with open(in_progress_preds_file, "w") as f:
                for pred in preds_list:
                    pred = f"{pred}\n"
                    f.write(pred)

            self.log(metrics)

            metrics['predictions'] = preds_list



        else:
            metrics = output.metrics

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples=None, ignore_keys=None,
                metric_key_prefix: str = "test", output_save_dir=None, tokenizer=None):
        self._max_length = self.args.generation_max_length
        self._num_beams = self.args.generation_num_beams
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        predict_examples = self.eval_examples if predict_examples is None else predict_examples
        try:
            output = eval_loop(
                predict_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )

        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None:
            return output

        predictions = self.post_process_function(predict_examples, output, tokenizer, output_save_dir)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=None)