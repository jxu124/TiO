# Author: Jie Xu
# Date: 4/2/2023

# builtin part
from io import BytesIO
from functools import wraps
from collections import Counter
import logging
import warnings
import math
import re
import random
from dataclasses import dataclass
from typing import Optional, Any, Union, List
import json
import os
import yaml

# 3rd part
from PIL import Image, ImageFile
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
    with open("/mnt/bn/hri-lq/datasets/hf-cache/test_images.json", "r") as f:
        test_images = set(json.load(f))
    with open('/mnt/bn/hri-lq/projects/OFA-Invig/config/invig_env.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    path_images = cfg['env']['path_images']

    @staticmethod
    def filter_exclude_test_images(global_image_id):
        # ds.filter(MapFunc.filter_exclude_test_images, input_columns=['global_image_id'])
        if type(global_image_id) is list:
            return [(i not in MapFunc.test_images) for i in global_image_id]
        elif type(global_image_id) is str:  # str
            return global_image_id not in MapFunc.test_images
        else:
            raise ValueError
    
    @staticmethod
    def load_image(image_path):
        if type(image_path) is dict:
            image_path['image'] = MapFunc.load_image(image_path['image_path'])['image']
            return image_path
        # ds.map(MapFunc.load_images, input_columns=['image_path'])
        for k, v in MapFunc.path_images.items():
            image_path = image_path.replace(k, v, 1)
        image = Image.open(image_path)
        return {'image': image}
    
## ====== refcoco ======

    @staticmethod
    def refcoco_grounding(features):
        """ 任务数据集：Grounding """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        context = random.choice(json.loads(features['texts']))
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
        image = features['image']
        # 处理文本
        caption = features['caption']
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理文本
        caption = features['caption']
        dialog = json.loads(features['dialog'])
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
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        # bbox = features.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
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

## ====== objects365 ======

    @staticmethod
    def objects365_grounding(features, style=None):
        """ 任务数据集：Grounding """
        known_cates = set(['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong', 'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak', 'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis '])
        # 处理图片
        image = features['image']
        # 生成bbox
        anns_info = json.loads(features['anns_info'])
        bboxes = np.asarray([i["bbox"] for i in anns_info])
        categorys = [i["category"] for i in anns_info]
        c, num = random.choice(Counter(categorys).most_common())
        mask = [i==c for i in categorys]
        bboxes = bboxes[np.where(mask)].tolist()
        bboxes.sort(key=lambda a: a[0] * 100 + a[1])
        sbboxes = [bbox_to_sbbox(bbox, *image.size) for bbox in bboxes]
        # 对话模式 - 个数不对
        num_less = num - random.randint(0, num-1)
        num_more = num + random.randint(1, 4)
        # 对话模式 - 负样例类别
        false_c = random.choice(list(known_cates - set(categorys)))
        c = c.replace("/", " or ")
        false_c = false_c.replace("/", " or ")
        
        # 处理文本
        src_candidate = [
            f" \n#instruction: point out the location of any {num_less} {c}.",
            f" \n#instruction: point out the location of any {num_more} {c}.",
            f" \n#instruction: where are {c} located?",
            f" \n#instruction: how many {c} are there? where are they located?",
            f" \n#instruction: how many {false_c} are there? where are they located?"
        ]
        tgt_candidate = [  # num_less
            f" there are {num} {c}. here are {num_less} of them.\n" + "\n".join([f"- region: {sbbox}" for sbbox in random.choices(sbboxes, k=num_less)]),
            f" there are only {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes]),
            f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes]),
            f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes]),
            f" there is no {false_c}.",
        ]
        if style is None:
            weights = [2, 2, 4, 2, 1]
            style = random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== end of TaskProcess ======


@dataclass
class DataCollatorForOFA(transformers.DataCollatorForSeq2Seq):
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[Any] = None
    padding: Optional[str] = "longest"
    max_src_length: Optional[int] = 256
    max_tgt_length: Optional[int] = 128
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
        from datasets.distributed import split_dataset_by_node
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

