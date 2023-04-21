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

@dataclass
class LoadImage():
    path_images: dict
    def __call__(self, features):
        if features.get('image_path') is not None:
            for k, v in self.path_images.items():
                features['image_path'] = features['image_path'].replace(k, v)
            features['image'] = Image.open(features['image_path'])
        return features


class TaskProcessor():
    @staticmethod
    def setup_task(dataset, task):
        return dataset.add_column("__task__", [task] * len(dataset))

    @staticmethod
    def auto_task(data: dict, **args):
        if "path_images" in args:
            data = LoadImage(path_images=args.pop('path_images'))(data)
        return getattr(TaskProcessor, data['__task__'])(data, **args)

    def dataset_wrapper(gttd_func):
        @wraps(gttd_func)
        def wrapper(data):
            # 处理图片
            image = data.get('image', None)
            if image is None:
                image = Image.open(data['image_path'])
            # 处理bbox
            w, h = image.size
            bbox = data.get('bbox', None)
            sbbox = bbox_to_sbbox(bbox, w, h) if bbox is not None else None
            caption = data.get("caption", "")
            assert len(data['dialog'][0]) == 2
            src_text, tgt_text = gttd_func(data['dialog'], sbbox, caption)
            src_text, tgt_text = src_text.lower(), tgt_text.lower()
            return src_text, tgt_text, image
        return wrapper

    @staticmethod
    @dataset_wrapper  # src_text, tgt_text, image = gttd_question(data)
    def question(dialog, sbbox=None, caption=""):
        """ 任务数据集：提问 """
        for _ in range(10):
            turn_left = 1 if dialog[0][0] != "" else 2  # 如果第一轮“question”为“”，那么不能把“”作为tgt_text
            turn = random.randint(min(turn_left, len(dialog)), len(dialog))  # 选一个轮数，来作为tgt_text
            context = []
            for t in range(turn-1):
                if dialog[t][0] != "":
                    context += [f"question: {dialog[t][0]}"]
                context += [f"answer: {dialog[t][1]}"]
            context = " ".join(context)
            question = dialog[turn-1][0]
            if not question.endswith("."):  # 消除ok, i see.
                break
        
        caption = caption + " " if caption else ""
        pattern = re.compile(r'^(is|do|does|did|will|would|are|am|can|could|may|might|have|has)$')
        ask_a_quesion = "ask a dichotomous closed-ended question" if pattern.match(question) is not None else "ask a question"
        src_candidate = [
            f" \n#instruction: {ask_a_quesion} to identify the target. \n#context: \"{caption}{context}\"",
            f" \n#instruction: can you provide which region the context describe does? \n#context: \"{caption}{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" not yet. {question}"
        ]
        style = random.choices([0, 1], weights=[3, 1])[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return src_text, tgt_text

    @staticmethod
    @dataset_wrapper
    def answer(dialog, sbbox=None, caption=""):
        """ 任务数据集：回答 """
        ed = len(dialog) if dialog[-1][1] != "" else max(1, len(dialog)-1)
        turn = random.randint(1, ed)
        context = []
        for t in range(turn-1):
            if dialog[t][0] != "":
                context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        caption = caption + " " if caption else ""
        according_to_the_region = " according to the region" if sbbox is not None else ""
        yes_or_no = " yes or no?" if answer.lower() in ["yes.", "no."] else ""  # "I'm not sure."
        region = f"\n#region: {sbbox}" if sbbox is not None else ""
        src_candidate = [
            f" \n#instruction: answer following{according_to_the_region}. {question}{yes_or_no}{region}\n#context: \"{caption}{context}\"",
        ]
        tgt_candidate = [
            f" {answer}"
        ]
        style = random.choices([0, ], weights=[1, ])[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return src_text, tgt_text

    @staticmethod
    @dataset_wrapper
    def grounding(dialog, sbbox=None, caption=""):
        """ 任务数据集：Grounding """
        assert sbbox is not None
        turn = len(dialog)
        context = []
        for t in range(turn):
            if dialog[t][0] != "":
                context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        
        caption = caption + " " if caption else ""
        src_candidate = [
            f" \n#instruction: which region does the context describe? \n#context: \"{caption}{context}\"",
            f" \n#instruction: can you provide which region the context describes? \n#context: \"{caption}{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" sure. region: {sbbox}"
        ]
        style = random.choices([0, 1], weights=[2, 1])[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return src_text, tgt_text
    
    @staticmethod
    @dataset_wrapper
    def grounding_for_guesswhat(dialog, sbbox=None, caption=""):
        """ 任务数据集：Grounding """
        assert sbbox is not None
        # 增强部分：对话乱序、对话随机删除
        turn = len(dialog)
        context = []
        for t in range(turn):
            if dialog[t][0] != "":
                context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        
        caption = caption + " " if caption else ""
        src_candidate = [
            f" \n#instruction: which region does the context describe? \n#context: \"{caption}{context}\"",
            f" \n#instruction: can you provide which region the context describes? \n#context: \"{caption}{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" sure. region: {sbbox}"
        ]
        style = 0  # random.choices([0, 1], weights=[2, 1])[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return src_text, tgt_text

    @staticmethod
    @dataset_wrapper
    def grounding_for_test(dialog, sbbox=None, caption=""):
        """ 任务数据集：Grounding """
        assert sbbox is not None
        turn = len(dialog)
        context = []
        for t in range(turn):
            if dialog[t][0] != "":
                context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        caption = f"{caption} " if caption else ""

        src_text = f" \n#instruction: which region does the context describe? \n#context: \"{caption}{context}\"".lower()
        tgt_text = f" region: {sbbox}".lower()
        return src_text, tgt_text

    @staticmethod
    @dataset_wrapper
    def question_legency(dialog, sbbox=None, caption=""):
        for _ in range(10):
            turn_left = 1 if dialog[0][0] != "" else 2  # 如果第一轮“question”为“”，那么不能把“”作为tgt_text
            turn = random.randint(min(turn_left, len(dialog)), len(dialog))  # 选一个轮数，来作为tgt_text
            context = []
            for t in range(turn-1):
                if dialog[t][0] != "":
                    context += [f"q: {dialog[t][0]}"]
                context += [f"a: {dialog[t][1]}"]
            context = " ".join(context).lower()
            question = dialog[turn-1][0].lower()
            if not question.endswith("."):
                break

        if context.startswith("a: "):
            context = context[3:]
        
        src_text = f"Output a question. {context}"
        tgt_text = f"q: {question}"
        return src_text, tgt_text
    
    @staticmethod
    @dataset_wrapper
    def answer_legency(dialog, sbbox=None, caption=""):
        ed = len(dialog) if dialog[-1][1] != "" else max(1, len(dialog)-1)
        turn = random.randint(1, ed)
        context = []
        for t in range(turn-1):
            if dialog[t][0] != "":
                context += [f"q: {dialog[t][0]}"]
            context += [f"a: {dialog[t][1]}"]
        context = " ".join(context).lower()
        question = dialog[turn-1][0].lower()
        answer = dialog[turn-1][1].lower()

        if context.startswith("a: "):
            context = context[3:]

        if sbbox is None:
            src_text = f"Output an answer. {context} q: {question}"
        else:
            src_text = f"Output an answer. Given {sbbox} . {context} q: {question}"
        tgt_text = f"a: {answer}"
        return src_text, tgt_text
    
    @staticmethod
    @dataset_wrapper
    def grounding_legency(dialog, sbbox=None, caption=""):
        assert sbbox is not None
        turn = len(dialog)
        context = [caption]
        for t in range(turn):
            if dialog[t][0] != "":
                context += [f"q: {dialog[t][0]}"]
            context += [f"a: {dialog[t][1]}"]
        context = " ".join(context).lower()
        # context = context.replace("a:", "g:", -1)  # 复现bug

        if context.startswith("a: "):
            context = context[3:]

        src_text = f"Output the region described. {context}"
        tgt_text = f"g: {sbbox}"
        return src_text, tgt_text

## ====== refcoco ======

    @staticmethod
    def refcoco_grounding(data):
        """ 任务数据集：Grounding """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        context = data['text']
        src_candidate = [
            f" \n#instruction: which region does the context describe? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
        ]
        style = 0
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return src_text, tgt_text, image

## ====== visdial ======

    @staticmethod
    def visdial_question(data, weights=None):
        """ question_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理文本
        caption = data['caption']
        dialog = data['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: let's talk about this picture."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}?"]
            context += [f"answer: {dialog[t][1]}."]
        context = " ".join(context)
        question = dialog[turn-1][0]
        
        src_candidate = [
            f" \n#instruction: ask a question. \n#caption: \"{caption}.\"\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
        ]
        if weights is None:
            weights = [1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 2
        return src_text, tgt_text, image
    
    @staticmethod
    def visdial_answer(data, weights=None):
        """ answer_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理文本
        caption = data['caption']
        dialog = data['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: let's talk about this picture."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}?"]
            context += [f"answer: {dialog[t][1]}."]
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        short_or_detail = "briefly" if len(answer) < 20 else "in detail"
        src_candidate = [
            f" \n#instruction: answer the question {short_or_detail}. \n#caption: \"{caption}.\"\n#context: \"{context}\"\n#question: \"{question}\"",
        ]
        tgt_candidate = [
            f" {answer}",
        ]
        if weights is None:
            weights = [1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        return src_text, tgt_text, image

## ====== invig ======

    @staticmethod
    def invig_question(data, weights=None):
        """ question_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        # bbox = data.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
            f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" not yet. {question}"
        ]
        if weights is None:
            weights = [9, 1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 2
        return src_text, tgt_text, image
    
    @staticmethod
    def invig_answer(data, weights=None):
        """ answer_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
            f" \n#instruction: answer the question according to the region and context. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\"",
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
        return src_text, tgt_text, image
    
    @staticmethod
    def invig_grounding(data, weights=None, test=False):
        """ grounding_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
        return src_text, tgt_text, image
    
## ====== guesswhat ======
    
    @staticmethod
    def guesswhat_question(data, weights=None):
        """ question_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        # bbox = data.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"query: guess what i want."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}"]
            context += [f"answer: {dialog[t][1]}"]
        context = " ".join(context)
        question = dialog[turn-1][0]
        
        src_candidate = [
            f" \n#instruction: ask a question. \n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" not yet. {question}"
        ]
        if weights is None:
            weights = [9, 1]
        style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 1, f"{image}, {question}, {len(dialog)}"
        return src_text, tgt_text, image
    
    @staticmethod
    def guesswhat_answer(data, weights=None):
        """ answer_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
        return src_text, tgt_text, image
    
    @staticmethod
    def guesswhat_grounding(data, weights=None):
        """ grounding_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
        return src_text, tgt_text, image

## ====== ofa tasks ======

    @staticmethod
    def invig_grounding_ofa(data, weights=None):
        """ grounding_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
        return src_text, tgt_text, image

    @staticmethod
    def guesswhat_answer_oracle(data, weights=None):
        """ answer_invig """
        # 处理图片
        image = data.get('image', None)
        # 处理bbox
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = data['dialog']
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
        return src_text, tgt_text, image

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
        src_texts = [f[0] for f in features]
        tgt_texts = [f[1] for f in features]
        patch_images = torch.stack([self.image_processor(f[2]) for f in features])
        
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
    def __init__(self, dataset, path_images: dict, tokenizer=None, image_processor=None, **args):
        # 分布式适配
        _, rank, world_size = world_info_from_env()
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        # self.dataset.set_transform()
        self.path_images = path_images
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.args = args
        
    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        if type(index) is tuple:
            raise NotImplementedError
        else:
            data = TaskProcessor.auto_task(data, path_images=self.path_images, **self.args)
        return data
    
    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.dataset.shuffle(seed=42+epoch)

    def collater(self, samples, pad_to_length=None):
        collate_fn = DataCollatorForOFA(tokenizer=self.tokenizer, image_processor=self.image_processor)
        return collate_fn(samples)
