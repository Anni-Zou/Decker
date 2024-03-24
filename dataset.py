# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" CREAK, CSQA2 fine-tuning: utilities to work with commonsense reasoning """

import sys
import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import tqdm
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import torch
from torch.utils.data.dataset import Dataset
import random
from random import choice
import numpy as np
logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)




@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of question.
        facts: list of str. The untokenized text of facts related with the question.
        concept_ids: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    example_id: str
    question: str
    filtered_facts: List[str]
    concept_ids: List[int]
    cpt_to_cpt: dict
    cpt_wiz_fct: dict
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    """
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    #fact_ids: List[List[int]]  
    #fact_mask: Optional[List[List[int]]]
    #fact_type: Optional[List[List[int]]]
    concept_ids: List[int]
    #cpt2cpt_info: List[List[int]]   #src, dst, (rel)
    #cpt2fct_info: List[List[int]]   #src, dst
    adj_cpt2cpt: List[List[int]]    #(n_nodes, n_nodes)
    adj_cpt2fct: List[List[int]]    #(n_nodes, n_nodes)
    graph_mask: List[int]           #(n_nodes)    
    #num_actual: int           #num_cpt, cptrel, fct_rel
    label: Optional[int]

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    contrast = "contrast"

class MyDataset(Dataset):
    """
    Dataset class for CREAK, CSQA2
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,  #'creak' / 'csqa2'
        max_seq_length: Optional[int] = None,
        max_cpt_num: Optional[int] = None,
        max_fct_num: Optional[int] = None,
        overwrite_cache = False,
        mode: Split = Split.train,
    ):
        self.features = []

        task_data_dir = os.path.join(data_dir, task)
        if not os.path.exists(task_data_dir):
            raise ValueError("Data directory ({}) does not exist!".format(task_data_dir))
        cached_features_file = os.path.join(
            task_data_dir,
            f"cached_features_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}_{max_cpt_num}_{max_fct_num}"
        )

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading {task} features from cached file {cached_features_file}")
                print(f"Loading {task} features from cached file {cached_features_file}")
                features = torch.load(cached_features_file)
            else:
                processor = processors[task]()
                label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(task_data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(task_data_dir)
                elif mode == Split.contrast:
                    examples = processor.get_contrast_examples(task_data_dir)
                else:
                    examples = processor.get_train_examples(task_data_dir)
                logger.info(f"Training examples: {len(examples)}")
                features = convert_examples_to_features(
                    examples, label_list, max_seq_length, max_cpt_num, max_fct_num,tokenizer
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(features, cached_features_file)
        self.features.extend(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        example = self.features[index]
        input_ids = example.input_ids
        attention_mask = example.attention_mask
        token_type_ids = example.token_type_ids
        concept_ids = example.concept_ids
        adj_cpt2cpt = example.adj_cpt2cpt
        adj_cpt2fct = example.adj_cpt2fct
        graph_mask = example.graph_mask
        label = example.label
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                "concept_ids":concept_ids, "adj_cpt2cpt":adj_cpt2cpt, "adj_cpt2fct":adj_cpt2fct, "graph_mask":graph_mask,
                "label": label}


'''
class MyCollator(object):
    def __call__(self, features: List[InputFeatures]) -> Dict[str,Any]:

        if not isinstance(features[0], dict):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        for k, v in first.items():
            if k not in ("labels","label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif k=='graph':
                    batch[k] = dgl.batch([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
                #### plus dgl batch
        return batch
'''




class DataProcessor:
    """Base class for data converters for commonsense reasoning data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class CSQA2Processor(DataProcessor):
    """Processor for the CSQA2 data set."""

    def get_train_examples(self, data_dir, task=None):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "train_graph_numnode20.jsonl")), "")

    def get_dev_examples(self, data_dir, task=None):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "dev_graph_numnode20.jsonl")), "")

    def get_test_examples(self, data_dir, task=None):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "test_graph_numnode20.jsonl")), None)

    def get_labels(self):
        """See base class."""
        return ["yes", "no"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels):
        """Creates examples for the training and dev sets."""
        if labels is None:
            examples = [
                InputExample(
                    example_id= str(idx),
                    question = line["question"],
                    facts = line["filtered_facts"],
                    concept_ids = line["concept_ids"],
                    cpt_to_cpt = line["cpt_to_cpt"],
                    cpt_wiz_fct = line["cpt_wiz_fct"],
                    label="yes",
                )
                for idx, line in enumerate(lines)
            ]
        else:
            examples = [
                InputExample(
                    example_id= str(idx),
                    question = line["question"],
                    facts = line["filtered_facts"],
                    concept_ids = line["concept_ids"],
                    cpt_to_cpt = line["cpt_to_cpt"],
                    cpt_wiz_fct = line["cpt_wiz_fct"],
                    label=line["ans"],
                )
                for idx, line in enumerate(lines)

            ]
        return examples

class CREAKProcessor(DataProcessor):
    """Processor for the CREAK data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "train_graph_numnode20.jsonl")), "")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "dev_graph_numnode20.jsonl")), "")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "test_graph_numnode20.jsonl")), None)

    def get_contrast_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} contrast")
        return self._create_examples(self._read_json(os.path.join(data_dir, "graph", "contrast_graph_numnode20.jsonl")), "")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels):
        """Creates examples for the training and dev sets."""
        if labels is None:
            examples = [
                InputExample(
                    example_id= str(idx),
                    question = line["question"],
                    facts = line["filtered_facts"],
                    concept_ids = line["concept_ids"],
                    cpt_to_cpt = line["cpt_to_cpt"],
                    cpt_wiz_fct = line["cpt_wiz_fct"],
                    label="true",
                )
                for idx, line in enumerate(lines)
            ]
        else:
            examples = [
                InputExample(
                    example_id= str(idx),
                    question = line["question"],
                    facts = line["filtered_facts"],
                    concept_ids = line["concept_ids"],
                    cpt_to_cpt = line["cpt_to_cpt"],
                    cpt_wiz_fct = line["cpt_wiz_fct"],
                    label=line["ans"],
                )
                for idx, line in enumerate(lines)

            ]
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    max_cpt_num: int,
    max_fct_num: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Load a data file into a list in `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        #print(ex_index)
        if ex_index % 1000 == 0:
            logger.info(f"Converting example {ex_index} of {len(examples)} into features")
        
        total_inputs = []
        question_inputs = tokenizer(example.question, max_length=max_seq_length, truncation=True, padding='max_length')
        #input_ids = question_inputs["input_ids"]
        #attention_mask = question_inputs["attention_mask"] if "attention_mask" in question_inputs else None
        #token_type_ids = question_inputs["token_type_ids"] if "token_type_ids" in question_inputs else None
        total_inputs.append(question_inputs) 
        for _, fact in enumerate(example.filtered_facts):
            text_a = fact
            inputs = tokenizer(text_a, None, add_special_tokens=True, max_length=max_seq_length, truncation='longest_first',
                padding='max_length', return_attention_mask=True, return_token_type_ids=True)
            
            #if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            #    logger.info(f"Attention! You are truncating tokens!!")

            total_inputs.append(inputs)


        input_ids = [x["input_ids"] for x in total_inputs]
        attention_mask = (
            [x["attention_mask"] for x in total_inputs] if "attention_mask" in total_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in total_inputs] if "token_type_ids" in total_inputs[0] else None
        )

        n_total_nodes = max_cpt_num+max_fct_num
        adj_cpt2cpt = np.zeros((n_total_nodes, n_total_nodes), dtype=int)
        adj_cpt2fct = np.zeros((n_total_nodes, n_total_nodes), dtype=int)
        graph_mask = [0]*n_total_nodes
        cpt2cpt_rel = len(example.cpt_to_cpt["src"])
        cpt2fct_rel = len(example.cpt_wiz_fct["src"])
        for idx in range(n_total_nodes):
            adj_cpt2cpt[idx][idx], adj_cpt2fct[idx][idx] = 1, 1
        for idx in range(cpt2cpt_rel):
            src, dst = example.cpt_to_cpt["src"][idx], example.cpt_to_cpt["dst"][idx]
            adj_cpt2cpt[src][dst] = 1
        for idx in range(cpt2fct_rel):
            src, dst = example.cpt_wiz_fct["src"][idx], example.cpt_wiz_fct["dst"][idx]
            adj_cpt2fct[src][dst], adj_cpt2fct[dst][src] = 1, 1
        adj_cpt2cpt = adj_cpt2cpt.tolist()
        adj_cpt2fct = adj_cpt2fct.tolist()

        concept_ids = example.concept_ids
        #num_actual = []
        cpts = concept_ids.copy()
        while -1 in cpts: cpts.remove(-1)
        num_actual = len(cpts)
        graph_mask[:num_actual+max_fct_num] = [1]*(num_actual+max_fct_num)
        

        #num_actual.append(truncate_pad_list(example.cpt_to_cpt["src"],max_cptrel_num)[1])
        #num_actual.append(truncate_pad_list(example.cpt_wiz_fct["src"],max_fctrel_num)[1])
        #cpt2cpt_info =[
        #        truncate_pad_list(example.cpt_to_cpt["src"],max_cptrel_num)[0], 
        #        truncate_pad_list(example.cpt_to_cpt["dst"],max_cptrel_num)[0], 
                #truncate_pad_list(example.cpt_to_cpt["rel"],max_cptrel_num)[0]
        #        ]
        #cpt2fct_info = [
        #        truncate_pad_list(example.cpt_wiz_fct["src"],max_fctrel_num)[0], 
        #        truncate_pad_list(example.cpt_wiz_fct["dst"],max_fctrel_num)[0]]

        label = label_map[example.label]
        features.append(
            InputFeatures(
                example_id = str(ex_index),
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                #fact_ids = fact_ids,
                #fact_mask = fact_mask,
                #fact_type = fact_type,
                concept_ids = concept_ids,
                adj_cpt2cpt = adj_cpt2cpt,
                adj_cpt2fct = adj_cpt2fct,
                graph_mask = graph_mask,
                #cpt2cpt_info = cpt2cpt_info,
                #cpt2fct_info = cpt2fct_info,
                #num_actual = num_actual,
                label=label
            )
        )
    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

def truncate_pad_list(list, slen):
    cur_len = len(list)
    if cur_len < slen:
        list += [0]*(slen-cur_len)
    elif cur_len > slen:
        list = list[:slen]
    return list, min(cur_len, slen)

processors = {"creak": CREAKProcessor,
              "csqa2": CSQA2Processor
              }

