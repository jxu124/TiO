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

# 3rd part
from PIL import Image, ImageFile
from datasets.distributed import split_dataset_by_node
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


class TaskProcessor():
    @staticmethod
    def setup_task(dataset, task):
        return dataset.add_column("_task", [task] * len(dataset))

    @staticmethod
    def auto_task(data):
        return getattr(TaskProcessor, data['_task'])(data)

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
    def question_legency(self, dialog, sbbox=None, caption=""):
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
    def answer_legency(self, dialog, sbbox=None, caption=""):
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


# ====== old legency dataset define ======


class DatasetWarper():
    def __init__(self, dataset, task):
        self.dataset = dataset
        task_map = {
            "question": TaskProcessor.question,
            "answer": TaskProcessor.answer,
            "grounding": TaskProcessor.grounding,
        }
        self.task_process = task_map[task]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        return self.task_process(data)

        # # 处理图片
        # image = data.get('image', None)
        # if image is None:
        #     image = Image.open(data['image_path'])
        # # 处理bbox
        # w, h = image.size
        # bbox = data.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, w, h) if bbox is not None else None
        # caption = data.get("caption", "")
        # src_text, tgt_text = self.get_text_pair_from_dialog(data['dialog'], sbbox, caption)
        
        # return src_text.lower(), tgt_text.lower(), image


# ====== ofa_fairseq 通用的 collate 函数 ======

def collate_ofa_fairseq(batch, tokenizer, image_processor):
    """
    collate_fn for OFA.
    完整的数据格式在OFA-fairseq中如下：
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "decoder_prompts": decoder_prompts,
        "target": target,
        "prefix_tokens": prefix_tokens,
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        "region_coords": region_coords
    }
    """
    # data in batch: (src_text, tgt_text, image)
    # batch = [{k: v[i] for k, v in batch.items()} for i in range(len(batch['_task']))]
    # batch = [TaskProcessor.auto_task(data) for data in batch]
    src_texts = [i[0] for i in batch]
    tgt_texts = [i[1] for i in batch]
    patch_images = torch.stack([image_processor(i[2]) for i in batch])
    
    # max_length longest
    src_inputs = tokenizer(src_texts, return_length=True, max_length=160, padding="longest", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=64, padding="longest", return_tensors="pt")
    src_inputs["labels"] = labels["input_ids"]

    src_tokens = src_inputs["input_ids"]
    src_lengths = src_inputs["length"]
    patch_masks = torch.tensor([True] * len(src_lengths))
    prev_output_tokens = src_inputs["labels"][..., :-1]
    target = src_inputs["labels"][..., 1:]
    batch = {
        "net_input": {
            'src_tokens': src_tokens,
            "src_lengths": src_lengths,
            'patch_images': patch_images,
            'patch_masks': patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }
    # 添加其他内容
    batch["nsentences"] = len(batch)
    batch["ntokens"] = src_lengths.sum().item()
    return batch


class OFADialogDataset(FairseqDataset):
    def __init__(self, dataset, tokenizer, image_processor):
        # 分布式适配
        local_rank, rank, world_size = world_info_from_env()
        raw_dataset = dataset
        dataset = split_dataset_by_node(raw_dataset, rank=rank, world_size=world_size)
        # 适配filedataset.py
        dataset.get_total_row_count = lambda: len(raw_dataset)
        dataset._seek = lambda offset=0: None
        self.dataset = dataset
        self.tokenizer, self.image_processor = tokenizer, image_processor
        
    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        if type(index) is tuple:
            raise NotImplemented
        else:
            data = TaskProcessor.auto_task(data)
        return data
    
    def __len__(self):
        return len(self.dataset)

    def collater(self, samples, pad_to_length=None):
        return collate_ofa_fairseq(samples, self.tokenizer, self.image_processor)
