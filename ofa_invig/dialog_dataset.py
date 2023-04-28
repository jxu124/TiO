# Author: Jie Xu
# Date: 4/2/2023

# builtin part
from io import BytesIO
from functools import wraps
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Any, Union, List
import argparse
import logging
import warnings
import math
import re
import random
import json
import os
import yaml

# 3rd part
from PIL import Image, ImageFile
from datasets import load_from_disk, Features, Value
from datasets.distributed import split_dataset_by_node
import transformers
import datasets
import numpy as np
import torch
import base64

# fairseq part
from fairseq.data import FairseqDataset

# invig module
from .common import bbox_to_sbbox, world_info_from_env, get_processor


logger = logging.getLogger(__name__)

"""
使用方法：(例子是用三个数据集构建了三个任务数据集然后合并为一个总的数据集)
"""

class MapFunc():
    cfg: dict = {}
    path_images: dict = {}
    test_images: set = set()
    get_processor: Any = None

    @staticmethod
    def load_config(path_yaml):
        with open(path_yaml) as f:
            MapFunc.cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(MapFunc.cfg['env']['path_exclude_images'], "r") as f:
            MapFunc.test_images = set(json.load(f))
        MapFunc.path_images = MapFunc.cfg['env']['path_images']
        MapFunc.processor = get_processor(pretrain_path=MapFunc.cfg['env']['path_tokenizer'], resolution=512)

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
    def load_image(features):
        # ds.map(MapFunc.load_images, input_columns=['image_path'])
        image_path = features['image_path']
        for k, v in MapFunc.path_images.items():
            image_path = image_path.replace(k, v, 1)
        return {**features, 'image': Image.open(image_path)}

    @staticmethod
    def imshow_bboxes(image, bboxes, colors="white"):
        from PIL import Image
        import mmcv
        image = mmcv.imshow_bboxes(np.asarray(image), bboxes, colors, show=False)
        return Image.fromarray(image)

    @staticmethod
    def make_src_text(instruction: str,
                      context: Optional[Union[list, str]] = None,
                      region: Optional[str] = None,
                      human_first: Optional[bool] = False) -> str:
        if human_first:
            speakers = ['human', 'agent']
        else:
            speakers = ['agent', 'human']
        if context is None:
            context = ""
        elif type(context) is list:
            context = " ".join([f"{speakers[i%2]}: {c}" for i, c in enumerate(context)])
        src_text = f" \n#instruction: {instruction}\n#context: \"{context}\""
        if region is not None:
            src_text = src_text + f"\n#region: {region}"
        return src_text

    @staticmethod
    def make_text_pair(dialog, min_turn: Optional[int] = None, src_sbbox: Optional[str] = None, human_first: Optional[bool] = False):
        """
        min_turn - None: 包括所有对话 (int): 随机n轮次之后的对话部分;
        human_first - 默认为False, 指的是 instruction 之后的第一句话是人类说的还是智能体说的;
        #instruction: <第一句话>
        #context: <第二句话到在倒数第二句>
        #region: <单独给的>
        tgt_text: <最后一句话>
        """
        # max_turn(human_first) - [3, 5, 7, 9, ...] -> [0, 1, 2, 3]
        # max_turn(agent_first) - [2, 4, 6, 8, ...] -> [0, 1, 2, 3]
        max_turn = (len(dialog) - 1) // 2 - 1 if human_first else len(dialog) // 2 - 1
        if min_turn is None:
            turn = max_turn
        else:
            # assert min_turn <= max_turn, f"【对话轮数错误】min_turn: {min_turn}, max_turn: {max_turn}"
            if min_turn > max_turn:
                warnings.warn(f"【对话轮数错误】min_turn: {min_turn}, max_turn: {max_turn}")
            turn = random.randint(min(min_turn, max_turn), max_turn)
        if human_first:
            # turn in [2, 4, 6, 8, ...] # turn * 2 + 2
            i = turn * 2 + 2
        else:
            # turn in [1, 3, 5, 7, ...] # turn * 2 + 1
            i = turn * 2 + 1
        instruction = dialog[0]
        context = dialog[1:i]
        tgt_text = dialog[i]
        src_text = MapFunc.make_src_text(instruction, context, src_sbbox, human_first).replace("  ", " ")
        tgt_text = (" " + tgt_text.strip()).replace("  ", " ")
        return src_text, tgt_text

# ====== refcoco ======

    @staticmethod
    def refcoco_caption(features, style=None):
        """ 任务数据集：Grounding """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        caption = random.choice(features['captions'])
        dialogs = [["describe the region briefly.", caption]]
        text_candidate = [MapFunc.make_text_pair(d, src_sbbox=sbbox) for d in dialogs]

        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def refcoco_grounding(features, style=None):
        """ 任务数据集：Grounding """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        caption = random.choice(features['captions'])
        dialogs = [["which region does the following text describe?", caption, f"region: {sbbox}"]]
        text_candidate = [MapFunc.make_text_pair(d, human_first=True) for d in dialogs]

        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== visdial_question ======

    @staticmethod
    def visdial_question(features, style=None):
        """ question_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0] + '?', i[1] + '.']]
        dialogs = [['ask a question based on the image.', *dialog]]
        text_candidate = [MapFunc.make_text_pair(d, 0) for d in dialogs]

        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def visdial_answer(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0] + '?', i[1] + '.']]
        dialogs = [['answer the question.', *dialog]]
        text_candidate = [MapFunc.make_text_pair(d, 0, human_first=True) for d in dialogs]

        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def visdial_caption(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        caption = features['caption']
        dialogs = [['what does the image show?', caption],
                   ['describe the image briefly.', caption]]
        text_candidate = [MapFunc.make_text_pair(d) for d in dialogs]

        weights = [1, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== invig ======

    @staticmethod
    def invig_question(features, style=None):
        """ question_invig """
        # 处理图片
        image = features.get('image', None)
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i][1:]
        if "?" not in dialog[-2] and len(dialog) > 3:  # 消除ok, i see.
            dialog = dialog[:-2]
        dialogs = [["ask a question to guess what i want.", *dialog],
                   ["can you specify which region the context describes?", *dialog]]
        text_candidate = [MapFunc.make_text_pair(d, 0, human_first=True) for d in dialogs]

        weights = [9, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " not yet." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def invig_answer(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i][1:]
        if "?" not in dialog[-2] and len(dialog) > 3:  # 消除ok, i see.
            dialog = dialog[:-2]
        dialogs = [["answer the question based on the region.", *dialog],
                   ["state your query about the region.", dialog[0]]]
        text_candidate = [MapFunc.make_text_pair(dialogs[0], 1, src_sbbox=sbbox), MapFunc.make_text_pair(dialogs[1], 0, src_sbbox=sbbox)]

        weights = [9, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        # tgt_text = "not yet. " + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def invig_grounding(features, style=None):
        """ grounding_invig """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0], i[1]]][1:]
        if "?" not in dialog[-2] and len(dialog) > 2:  # 消除ok, i see.
            dialog = dialog[:-2]
        dialogs = [["which region does the context describe?", *dialog, f" region: {sbbox}"],
                   ["can you specify which region the context describes?", *dialog, f" region: {sbbox}"],]
        text_candidate = [MapFunc.make_text_pair(d, human_first=True) for d in dialogs]

        weights = [9, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " sure." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== guesswhat ======

    @staticmethod
    def guesswhat_question(features, style=None):
        """ question_invig """
        # 处理图片
        image = features['image']
        # 处理bbox
        # bbox = features.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialogs = [["ask a close-ended question to guess what i want."] + dialog,
                   ["can you specify which region the context describes?"] + dialog]
        text_candidate = [MapFunc.make_text_pair(d, 0) for d in dialogs]

        weights = [9, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " not yet." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def guesswhat_answer(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialogs = [["answer the question based on the region with yes or no."] + dialog]
        text_candidate = [MapFunc.make_text_pair(d, 0, human_first=True, src_sbbox=sbbox) for d in dialogs]

        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        # tgt_text = "not yet. " + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def guesswhat_grounding(features, style=None):
        """ grounding_invig """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0], i[1]]]
        dialogs = [["which region does the context describe?"] + dialog + [f" region: {sbbox}"],
                   ["can you specify which region the context describes?"] + dialog + [f" region: {sbbox}"],]
        text_candidate = [MapFunc.make_text_pair(d) for d in dialogs]

        weights = [9, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " sure." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== objects365 ======

    @staticmethod
    def objects365_detection(features, style=None):
        """ 任务数据集：Grounding """
        raise NotImplementedError
        # openimages
        known_cates = set(['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong', 'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak', 'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis '])
        # 处理图片
        image = features['image']
        # 生成bbox
        anns_info = json.loads(features['anns_info'])
        bboxes = np.asarray([i["bbox"] for i in anns_info])
        categorys = [i["category"] for i in anns_info]
        c, num = random.choice(Counter(categorys).most_common())
        mask = [i == c for i in categorys]
        bboxes = bboxes[np.where(mask)].tolist()
        bboxes.sort(key=lambda a: a[0] * 100 + a[1])
        sbboxes = [bbox_to_sbbox(bbox, *image.size) for bbox in bboxes]
        # 对话模式 - 个数不对
        num_less = num - random.randint(0, num - 1)
        num_more = num + random.randint(1, 4)
        # 对话模式 - 负样例类别
        false_c = random.choice(list(known_cates - set(categorys)))
        c = c.replace("/", " or ")
        false_c = false_c.replace("/", " or ")

        # 处理文本
        src_candidate = [
            MapFunc.make_src_text(f"point out the location of any {num_less} {c}."),
            MapFunc.make_src_text(f"point out the location of any {num_more} {c}."),
            MapFunc.make_src_text(f"where are {c} located?"),
            MapFunc.make_src_text(f"how many {c} are there? where are they located?"),
            MapFunc.make_src_text(f"how many {false_c} are there? where are they located?"),
        ]
        tgt_candidate = [  # num_less
            f" there are {num} {c}. here are {num_less} of them.\n" + "\n".join([f"- region: {sbbox}" for sbbox in random.choices(sbboxes, k=num_less)]),
            f" there are only {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes]),
            f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes]),
            f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes]),
            f" there is no {false_c}.",
        ]
        weights = [2, 2, 3, 2, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== openimages_v1.2 ======

    @staticmethod
    def openimages_detection(features, style=None):
        """ 任务数据集：Grounding """
        # openimages
        known_cates = set(['Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ (Musical Instrument)', 'Cassette deck', 'Apple', 'Human eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Human beard', 'Bird', 'Parking meter', 'Traffic light', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll', 'Skull', 'Washing machine', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle', 'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones', 'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy', 'Screwdriver', 'Bicycle wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill (Tool)', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Brown bear', 'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot', 'Person', 'Bow and arrow', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat (Animal)', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Vehicle registration plate', 'Lantern', 'Toaster', 'Flashlight', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors', 'Stairs', 'Computer keyboard', 'Printer', 'Traffic sign', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock', 'Fire hydrant', 'Land vehicle', 'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Medical equipment', 'Cattle', 'Cello', 'Jet ski', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat', 'Bronze sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Computer mouse', 'Cookie', 'Office building', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Computer monitor', 'Box', 'Stapler', 'Christmas tree', 'Cowboy hat', 'Hiking equipment', 'Studio couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Human mouth', 'Dairy Product', 'Dice', 'Oven', 'Dinosaur', 'Ratchet (Device)', 'Couch', 'Cricket ball', 'Winter melon', 'Spatula', 'Whiteboard', 'Pencil sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Tin can', 'Mug', 'Tap', 'Harbor seal', 'Stretcher', 'Can opener', 'Goggles', 'Human body', 'Roller skates', 'Coffee cup', 'Cutting board', 'Blender', 'Plumbing fixture', 'Stop sign', 'Office supplies', 'Volleyball (Ball)', 'Vase', 'Slow cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Paper towel', 'Personal care', 'Food', 'Sun hat', 'Tree house', 'Flying disc', 'Skirt', 'Gas stove', 'Salt and pepper shakers', 'Mechanical fan', 'Face powder', 'Fax', 'Fruit', 'French fries', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'French horn', 'Window blind', 'Human foot', 'Golf cart', 'Jacket', 'Egg (Food)', 'Street light', 'Guitar', 'Pillow', 'Human leg', 'Isopod', 'Grape', 'Human ear', 'Power plugs and sockets', 'Panda', 'Giraffe', 'Woman', 'Door handle', 'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Baseball bat', 'Baseball glove', 'Mixing bowl', 'Marine invertebrates', 'Kitchen utensil', 'Light switch', 'House', 'Horse', 'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed', 'Adhesive tape', 'Harp', 'Sandal', 'Bicycle helmet', 'Saucer', 'Harpsichord', 'Human hair', 'Heater', 'Harmonica', 'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Drinking straw', 'Insect', 'Hair dryer', 'Kitchenware', 'Indoor rower', 'Invertebrate', 'Food processor', 'Bookcase', 'Refrigerator', 'Wood-burning stove', 'Punching bag', 'Common fig', 'Cocktail shaker', 'Jaguar (Animal)', 'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet', 'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Bottle opener', 'Lynx', 'Lavender (Plant)', 'Lighthouse', 'Dumbbell', 'Human head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 'Billiard table', 'Mammal', 'Mouse', 'Motorcycle', 'Musical instrument', 'Swim cap', 'Frying pan', 'Snowplow', 'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Waffle iron', 'Milk', 'Ring binder', 'Plate', 'Mobile phone', 'Baked goods', 'Mushroom', 'Crutch', 'Pitcher (Container)', 'Mirror', 'Personal flotation device', 'Table tennis racket', 'Pencil case', 'Musical keyboard', 'Scoreboard', 'Briefcase', 'Kitchen knife', 'Nail (Construction)', 'Tennis ball', 'Plastic bag', 'Oboe', 'Chest of drawers', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Hair spray', 'Sports equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Infant bed', 'Polar bear', 'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Digital clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'High heels', 'Red panda', 'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Submarine sandwich', 'Snowboard', 'Sword', 'Picture frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine', 'Scorpion', 'Segway', 'Training bench', 'Snake', 'Coffee table', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Picnic basket', 'Cooking spray', 'Trousers', 'Bowling equipment', 'Football helmet', 'Truck', 'Measuring cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Paper cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale', 'Zebra', 'Auto part', 'Jug', 'Pizza cutter', 'Cream', 'Monkey', 'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Toilet paper', 'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Wine glass', 'Countertop', 'Tablet computer', 'Waste container', 'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard', 'Axe', 'Hand dryer', 'Soap dispenser', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm tree', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Garden Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 'Horizontal bar', 'Convenience store', 'Bomb', 'Bench', 'Ice cream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange', 'Antelope', 'Beaker', 'Moths and butterflies', 'Window', 'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash (Plant)', 'Racket', 'Human face', 'Human arm', 'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Common sunflower', 'Microwave oven', 'Honeycomb', 'Marine mammal', 'Sea lion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Grinder', 'Spice rack', 'Light bulb', 'Corded phone', 'Sports uniform', 'Tennis racket', 'Wall clock', 'Serving tray', 'Kitchen & dining room table', 'Dog bed', 'Cake stand', 'Cat furniture', 'Bathroom accessory', 'Facial tissue holder', 'Pressure cooker', 'Kitchen appliance', 'Tire', 'Ruler', 'Luggage and bags', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band-aid', 'Animal', 'Bell pepper', 'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Human nose', 'Pen', 'Ant', 'Car', 'Aircraft', 'Human hand', 'Skunk', 'Teddy bear', 'Watermelon', 'Cantaloupe', 'Dishwasher', 'Flute', 'Balance beam', 'Sandwich', 'Shrimp', 'Sewing machine', 'Binoculars', 'Rays and skates', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe', 'Remote control', 'Wheelchair', 'Rugby ball', 'Armadillo', 'Maracas', 'Helmet'])
        # 处理图片
        image = features['image']
        # 生成bbox
        anns_info = features['anns_info']
        bboxes = np.asarray([i["bbox"] for i in anns_info])
        categorys = [i["category"] for i in anns_info]
        c, num = random.choice(Counter(categorys).most_common())
        mask = [i == c for i in categorys]
        bboxes = bboxes[np.where(mask)].tolist()
        bboxes.sort(key=lambda a: a[0] * 100 + a[1])
        sbboxes = [bbox_to_sbbox(bbox) for bbox in bboxes]
        # 对话模式 - 个数不对
        num_less = num - random.randint(0, num - 1)
        num_more = num + random.randint(1, 4)
        # 对话模式 - 负样例类别
        false_c = random.choice(list(known_cates - set(categorys)))
        c = c.replace("/", " or ")
        false_c = false_c.replace("/", " or ")

        dialogs = [[f"point out the location of any {num_less} {c}.", f" there are {num} {c}. here are {num_less} of them.\n" + "\n".join([f"- region: {sbbox}" for sbbox in random.choices(sbboxes, k=num_less)])],
                   [f"point out the location of any {num_more} {c}.", f" there are only {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes])],
                   [f"where are {c} located?", f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes])],
                   [f"how many {c} are there? where are they located?", f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes])],
                   [f"how many {false_c} are there? where are they located?", f" there is no {false_c}."]]
        text_candidate = [MapFunc.make_text_pair(d) for d in dialogs]

        weights = [2, 2, 3, 2, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== cc_sbu_align ======

    @staticmethod
    def cc_sbu_align_caption(features, style=None):
        """ 任务数据集：Grounding """
        image = features['image']
        caption = features.get('caption')
        # 处理文本
        dialogs = [['what does the image show?', caption],
                   ["describe the image.", caption]]
        text_candidate = [MapFunc.make_text_pair(d) for d in dialogs]
        weights = [1, 1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_instruct_150k ======

    @staticmethod
    def llava_instruct_150k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialog = ["answer the question.", *dialog]
        text_candidate = [MapFunc.make_text_pair(dialog, 0, human_first=True)]
        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_conversation_58k ======

    @staticmethod
    def llava_conversation_58k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialog = ["answer the question.", *dialog]
        text_candidate = [MapFunc.make_text_pair(dialog, 0, human_first=True)]
        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_complex_reasoning_77k ======

    @staticmethod
    def llava_complex_reasoning_77k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        text_candidate = [MapFunc.make_text_pair(dialog)]
        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_detail_23k ======

    @staticmethod
    def llava_detail_23k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        text_candidate = [MapFunc.make_text_pair(dialog)]
        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== end of TaskProcess ======

@dataclass
class DataCollatorForOFA(transformers.DataCollatorForSeq2Seq):
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[Any] = None
    padding: Optional[str] = "longest"
    max_src_length: Optional[int] = 256
    max_tgt_length: Optional[int] = 160
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


# class OFADataset(FairseqDataset, torch.utils.data.Dataset):  # ~OFADataset
#     def __init__(self, dataset, tokenizer=None, image_processor=None, **args):
#         from datasets.distributed import split_dataset_by_node
#         # 分布式适配
#         _, rank, world_size = world_info_from_env()
#         self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
#         self.tokenizer, self.image_processor = tokenizer, image_processor
#         self.args = args

#     def __getitem__(self, index):
#         return self.dataset.__getitem__(index)

#     def __len__(self):
#         return len(self.dataset)

#     def set_epoch(self, epoch):
#         super().set_epoch(epoch)
#         self.dataset.shuffle(seed=42 + epoch)

#     def collater(self, samples, pad_to_length=None):
#         collate_fn = DataCollatorForOFA(tokenizer=self.tokenizer, image_processor=self.image_processor)
#         return collate_fn(samples)


class OFADataset(FairseqDataset):  # ~OFADataset
    def __init__(self, ds, tokenizer=None, image_processor=None, **args):
        _, rank, world_size = world_info_from_env()
        total_count = len(ds)
        self.ds = split_dataset_by_node(ds, rank, world_size)
        self.dataset = argparse.Namespace(**{'get_total_row_count': lambda: total_count, "_seek": lambda offset=0: None})
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.collater_args = args

    def __getitem__(self, idx):
        data = self.ds[idx]
        return getattr(MapFunc, data['__task__'])(MapFunc.load_image(data))

    def __len__(self):
        return len(self.ds)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(epoch)
        elif hasattr(self.ds, "shuffle"):
            self.ds.shuffle(epoch)

    def collater(self, samples, pad_to_length=None):
        collate_fn = DataCollatorForOFA(tokenizer=self.tokenizer,
                                        image_processor=self.image_processor,
                                        **self.collater_args)  # or max_length
        return collate_fn(samples)


def get_dataset(split):
    # 1 按照node切分数据集
    datasets_collection = []
    for i in MapFunc.cfg[split]:
        if i.get('streaming'):
            continue
        name = i['name']
        logger.info(f"loading {name}-{split} from {i['path']}")
        ds = datasets.load_from_disk(i['path'])[split]
        for task, prob in zip(i['tasks'], i['probs']):
            ds_task = ds.add_column("__task__", [task] * len(ds))\
                        .filter(MapFunc.filter_exclude_test_images, input_columns=['global_image_id'])
            datasets_collection += [(name, task, ds_task, prob)]

    # 2 按照比例组合数据
    tasks = [i[1] for i in datasets_collection]
    use_datasets = [i[2] for i in datasets_collection]
    probs = np.asarray([i[3] for i in datasets_collection])
    counts = np.asarray([len(ds) for ds in use_datasets])
    n_samples = (probs * counts).astype(int)
    total_count = sum(n_samples) // 2048 * 2048 if split == 'train' else 2048
    n_samples[-1] = total_count - sum(n_samples[:-1])
    # DEBUG: n_samples = [256] * len(n_samples)
    use_datasets = [ds.shuffle().select(range(n)) for ds, n in zip(use_datasets, n_samples)]

    # 3 构造fairseq_dataset，送进去预处理函数
    max_src_length, max_tgt_length = MapFunc.cfg['hparam']['max_src_length'], MapFunc.cfg['hparam']['max_tgt_length']
    dataset = datasets.concatenate_datasets(use_datasets)
    dataset = OFADataset(dataset, *MapFunc.processor, max_src_length=max_src_length, max_tgt_length=max_tgt_length)

    # 4 打印logs
    logger.info(f"load {split} data: {len(use_datasets)} ({len(dataset)}/{total_count} samples) dataset(s)")
    logger.info("Tasks: " + ", ".join([f'{t}({n})' for t, n in zip(tasks, n_samples)]))

    return dataset
