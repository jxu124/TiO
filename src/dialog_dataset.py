# Author: Jie Xu
# Date: 4/2/2023

# builtin part
from io import BytesIO
from functools import wraps
import logging
import warnings
import math
import re
import random
from dataclasses import dataclass
from typing import Optional, Any, Union, List
import json
import yaml

# 3rd part
from PIL import Image, ImageFile
from datasets.distributed import split_dataset_by_node
import transformers
import datasets
import numpy as np
import torch
import base64

# fairseq part
from fairseq.data import FairseqDataset

# invig module
from .common import bbox_to_sbbox, world_info_from_env


"""
使用方法：(例子是用三个数据集构建了三个任务数据集然后合并为一个总的数据集)
from datasets import interleave_datasets
import datasets

image_processor = ...

# ds_1, ds_2, ds_3 to build different tasks
ds_question = interleave_datasets([ds_1, ds_3], probabilities=[0.5, 0.5], seed=42, stopping_strategy="all_exhausted")
ds_question = QuestionDataset(ds_question, image_processor)
ds_answer = interleave_datasets([ds_1, ds_3], probabilities=[0.5, 0.5], seed=42, stopping_strategy="all_exhausted")
ds_answer = AnswerDataset(ds_answer, image_processor)
ds_gnding = interleave_datasets([ds_2, ds_3], probabilities=[0.5, 0.5], seed=42, stopping_strategy="all_exhausted")
ds_gnding = AnswerDataset(ds_gnding, image_processor)

# tasks to build a union dataset
ds_all = interleave_datasets([ds_question, ds_answer, ds_gnding], probabilities=[0.33, 0.33, 0.33], seed=42, stopping_strategy="all_exhausted")
"""


class MapFunc():
    with open("/mnt/bn/hri-lq/datasets/hf/test_images.json", "r") as f:
        test_images = set(json.load(f))
    with open('/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_4ds.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    path_images = config['path_images']

    @staticmethod
    def filter_exclude_test_images(features):
        return features['image_path'] not in MapFunc.test_images
    
    @staticmethod
    def map_read_images(features):
        if features.get('image_path') is not None:
            for k, v in MapFunc.config['path_images'].items():
                features['image_path'] = features['image_path'].replace(k, v)
            features['image'] = Image.open(features['image_path'])
        return features

## ====== refcoco ======

    @staticmethod
    def refcoco_grounding(features):
        """ 任务数据集：Grounding """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        context = random.choice(features['texts'])
        src_candidate = [
            f" \n#instruction: which region does the context describe? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
        ]
        style = 0
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== visdial ======

    @staticmethod
    def visdial_question(features, weights=None):
        """ question_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理文本
        caption = features['caption']
        dialog = features['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: let's talk about the picture."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}?"]
            context += [f"answer: {dialog[t][1]}."]
        context = " ".join(context)
        question = dialog[turn-1][0]
        
        src_candidate = [
            f" \n#instruction: ask a question. \n#caption: \"{caption}.\"\n#context: \"{context}\"",
            f" \n#instruction: ask a question. \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" {question}",
        ]
        if weights is None:
            weights = [0.3, 0.7]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 2
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def visdial_answer(features, weights=None):
        """ answer_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理文本
        caption = features['caption']
        dialog = features['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: let's talk about the picture."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}?"]
            context += [f"answer: {dialog[t][1]}."]
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        short_or_detail = "briefly" if len(answer) < 20 else "in detail"
        src_candidate = [
            f" \n#instruction: answer the question {short_or_detail}. \n#caption: \"{caption}.\"\n#context: \"{context}\"\n#question: \"{question}\"",
            f" \n#instruction: answer the question {short_or_detail}. \n#context: \"{context}\"\n#question: \"{question}\"",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
        ]
        if weights is None:
            weights = [0.3, 0.7]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== invig ======

    @staticmethod
    def invig_question(features):
        """ question_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        # bbox = features.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        turn_right = len(dialog) - 1 if "?" not in dialog[-1][0] else len(dialog)  # 消除ok, i see.
        turn = random.randint(2, max(2, turn_right))  # 选一个轮数，来作为tgt_text
        context = [f"query: {dialog[0][1]}"]
        for t in range(1, turn-1):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        question = dialog[turn-1][0]
        
        src_candidate = [
            f" \n#instruction: ask a question. \n#context: \"{context}\"",
            f" \n#instruction: ask a new question. \n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" {question}",
            f" not yet. {question}"
        ]
        if turn <= 3:
            weights = [9, 0, 1]
        else:
            weights = [4, 5, 1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 2
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def invig_answer(features, weights=None):
        """ answer_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        turn_right = len(dialog) - 1 if "?" not in dialog[-1][0] else len(dialog)  # 消除ok, i see.
        turn = random.randint(2, max(2, turn_right))  # 选一个轮数，来作为tgt_text
        context = [f"query: {dialog[0][1]}"]
        for t in range(1, turn-1):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        src_candidate = [
            f" \n#instruction: answer the question according to the region. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\"",
            f" \n#instruction: say a word about the region. \n#region: {sbbox}",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {dialog[0][1]}",
        ]
        if weights is None:
            weights = [8, 2]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def invig_grounding(features, weights=None, test=False):
        """ grounding_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        context = [f"query: {dialog[0][1]}"]
        for t in range(1, len(dialog)):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        src_candidate = [
            f" \n#instruction: which region does the context describe? \n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" sure. region: {sbbox}",
        ]
        if weights is None:
            weights = [9, 1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        # style = 0 if test else style
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and sbbox
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
## ====== guesswhat ======
    
    @staticmethod
    def guesswhat_question(features):
        """ question_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        # bbox = features.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: guess what i want."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        question = dialog[turn-1][0]
        
        src_candidate = [
            f" \n#instruction: ask a question. \n#context: \"{context}\"",
            f" \n#instruction: ask a new question. \n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" {question}",
            f" not yet. {question}"
        ]
        if turn <= 3:
            weights = [9, 0, 1]
        else:
            weights = [4, 5, 1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 1, f"{image}, {question}, {len(dialog)}"
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def guesswhat_answer(features, weights=None):
        """ answer_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: guess what i want."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        src_candidate = [
            f" \n#instruction: answer the question with yes or no. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\"",
            f" \n#instruction: answer the question with a simple yes or no. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\"",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
        ]
        if weights is None:
            weights = [8, 2]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def guesswhat_grounding(features, weights=None):
        """ grounding_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        context = [f"query: guess what i want."]
        for t in range(0, len(dialog)):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        src_candidate = [
            f" \n#instruction: which region does the context describe? \n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" sure. region: {sbbox}",
        ]
        if weights is None:
            weights = [9, 1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and sbbox
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== ofa tasks ======

    @staticmethod
    def invig_grounding_ofa(features, weights=None):
        """ grounding_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        context = [f"query: {dialog[0][1]}"]
        for t in range(1, len(dialog)):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        src_candidate = [
            f" which region does the text \" {context} \" describe?",
        ]
        tgt_candidate = [
            f" {sbbox}",
        ]
        style = 0
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and sbbox
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def guesswhat_answer_oracle(features, weights=None):
        """ answer_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理bbox
        bbox = features.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = f"query: guess what i want."
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        src_candidate = [
            f" \n#instruction: answer the question with yes or no. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\"",
        ]
        tgt_candidate = [
            f" {answer}",
        ]
        style = 0
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== end of TaskProcess ======


@dataclass
class DataCollatorForOFA(transformers.DataCollatorForSeq2Seq):
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[Any] = None
    padding: Optional[str] = "longest"
    max_src_length: Optional[int] = 256
    max_tgt_length: Optional[int] = 64
    return_tensors: Optional[str] = "pt"
    
    def __call__(self, features, return_tensors=None):
        src_texts = [f['src_text'] for f in features]
        tgt_texts = [f['tgt_text'] for f in features]
        patch_images = torch.stack([self.image_processor(f['image']) for f in features])
        
        # max_length longest
        inputs = self.tokenizer(text=src_texts, return_length=True, max_length=self.max_src_length, 
                                padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(text=tgt_texts, max_length=self.max_tgt_length, 
                                    padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        # labels = self.tokenizer(text_target=tgt_texts, max_length=self.max_tgt_length, 
        #                         padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        patch_masks = torch.tensor([True] * len(patch_images))
        prev_output_tokens = labels.input_ids[..., :-1]
        target = labels.input_ids[..., 1:]
        features = {
            "net_input": {
                'src_tokens': inputs.input_ids,
                "src_lengths": inputs.length,
                'patch_images': patch_images,
                'patch_masks': patch_masks,
                "prev_output_tokens": prev_output_tokens
            },
            "target": target,
            # 仅fairseq训练需要
            "nsentences": len(inputs.length),
            "ntokens": inputs.length.sum().item()
        }
        return features


class OFADataset(FairseqDataset, torch.utils.data.Dataset):  # ~OFADataset
    def __init__(self, dataset, tokenizer=None, image_processor=None, **args):
        # 分布式适配
        _, rank, world_size = world_info_from_env()
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.args = args
        
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.dataset.shuffle(seed=42+epoch)

    def collater(self, samples, pad_to_length=None):
        collate_fn = DataCollatorForOFA(tokenizer=self.tokenizer, image_processor=self.image_processor)
        return collate_fn(samples)


@dataclass
class GenSampler():
    generators: list
    weights: list
    def __call__(self):
        while True:
            try:
                yield next(random.choices(self.iters, weights=self.weights)[0])
            except AttributeError:
                self.init_iter()
                yield next(random.choices(self.iters, weights=self.weights)[0])
            except StopIteration:
                self.init_iter()
                yield next(random.choices(self.iters, weights=self.weights)[0])
                # raise StopIteration
    def init_iter(self):
        self.iters = [iter(g) for g in self.generators]

