#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import transformers
import dgl
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.integrations import WandbCallback, rewrite_logs

from model.modeling_v3 import MyConFact_RGCN
from dataset import MyDataset, Split

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: Optional[str] = field(default="creak",
                                metadata={"help": "Task name"})
    data_dir: Optional[str] = field(default="./data/",
                                    metadata={"help": "Location of data"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_fact_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input fact length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_cptrel_num: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_fctrel_num: Optional[int] = field(
        default=20,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    ## Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    ## Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
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


    ## Set seed before initializing model.
    set_seed(training_args.seed)
    dgl.seed(training_args.seed)


    ## Setting up model and tokenizer configs
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = MyConFact_RGCN.from_pretrained(
        model_args.model_name_or_path,
        config = config,
        cache_dir = model_args.cache_dir,
        revision = model_args.model_revision,
        use_auth_token = True if model_args.use_auth_token else None
    )   ## tobe checked


    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    ## Get datasets
    if training_args.do_train:
        train_dataset = MyDataset(
            data_dir = data_args.data_dir,
            tokenizer = tokenizer,
            task = data_args.task,
            max_seq_length = max_seq_length,
            max_fact_length = data_args.max_fact_length,
            max_cptrel_num = data_args.max_cptrel_num,
            max_fctrel_num = data_args.max_fctrel_num,
            overwrite_cache = data_args.overwrite_cache,
            mode = Split.train
            )

    if training_args.do_eval:
        eval_dataset = MyDataset(
            data_dir = data_args.data_dir,
            tokenizer = tokenizer,
            task = data_args.task,
            max_seq_length = max_seq_length,
            max_fact_length = data_args.max_fact_length,
            max_cptrel_num = data_args.max_cptrel_num,
            max_fctrel_num = data_args.max_fctrel_num,
            overwrite_cache = data_args.overwrite_cache,
            mode = Split.dev
            )

    if training_args.do_predict:
        test_dataset = MyDataset(
            data_dir = data_args.data_dir,
            tokenizer = tokenizer,
            task = data_args.task,
            max_seq_length = max_seq_length,
            max_fact_length = data_args.max_fact_length,
            max_cptrel_num = data_args.max_cptrel_num,
            max_fctrel_num = data_args.max_fctrel_num,
            overwrite_cache = data_args.overwrite_cache,
            mode = Split.test
            )
        if data_args.task == 'creak':
            contra_dataset = MyDataset(
                data_dir = data_args.data_dir,
                tokenizer = tokenizer,
                task = data_args.task,
                max_seq_length = max_seq_length,
                max_fact_length = data_args.max_fact_length,
                max_cptrel_num = data_args.max_cptrel_num,
                max_fctrel_num = data_args.max_fctrel_num,
                overwrite_cache = data_args.overwrite_cache,
                mode = Split.contrast
                )


    ## Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    ## Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )


    ## Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["my_metrics"])
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    ## Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test Set Prediction
    if training_args.do_predict:
        logger.info("*** Predict: Test ***")
        predictions, labels, metrics = trainer.predict(test_dataset)

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if data_args.task == 'creak':
            label2id = {'true': 0, 'false': 1}
            id2label = {v: k for k, v in label2id.items()}
        elif data_args.task == 'csqa2':
            label2id = {'yes': 0, 'no': 1}
            id2label = {v: k for k, v in label2id.items()}

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir,
                                           "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = id2label[item]
                    writer.write(f"{index}\t{item}\n")

        logger.info("*** Predict: Contrast ***")
        predictions, labels, metrics = trainer.predict(contra_dataset)

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                contra_dataset)
        )
        metrics["contra_samples"] = min(max_predict_samples,
                                        len(contra_dataset))

        trainer.log_metrics("contra", metrics)
        trainer.save_metrics("contra", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir,
                                           "contra_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = id2label[item]
                    writer.write(f"{index}\t{item}\n")


    


if __name__ == "__main__":
    main()
