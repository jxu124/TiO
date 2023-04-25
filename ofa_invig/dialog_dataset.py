# Author: Jie Xu
# Date: 4/2/2023

# builtin part
from io import BytesIO
from functools import wraps
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Any, Union, List
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
"""

class MapFunc():
    cfg: dict = {}
    path_images: dict = {}
    test_images: set = set()

    @staticmethod
    def load_config(path_yaml):
        with open(path_yaml) as f:
            MapFunc.cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(MapFunc.cfg['env']['path_exclude_images'], "r") as f:
            MapFunc.test_images = set(json.load(f))
        MapFunc.path_images = MapFunc.cfg['env']['path_images']

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
    
## ====== refcoco ======

    @staticmethod
    def refcoco_caption(features, style=None):
        """ 任务数据集：Grounding """
        # 处理图片
        image = features['image']
        # 处理bbox
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        caption = random.choice(json.loads(features['texts']))
        src_candidate = [
            f" \n#instruction: describe the region with a phrase.\n#region: {sbbox}",
        ]
        tgt_candidate = [
            f" {caption}",
        ]
        weights = [1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
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
        caption = random.choice(json.loads(features['texts']))
        context = f"human: {caption}"
        src_candidate = [
            f" \n#instruction: which region does the context describe?\n#context: {caption}",
            f" \n#instruction: which region does the context describe? {caption}",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" region: {sbbox}",
        ]
        weights = [0.3, 0.7]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== visdial ======

    @staticmethod
    def visdial_question(features, style=None):
        """ question_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = json.loads(features['dialog'])
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = []
        for t in range(turn-1):
            context += [f"agent: {dialog[t][0]}?"]  # question
            context += [f"human: {dialog[t][1]}."]  # answer
        context = " ".join(context)
        # caption = features['caption']
        question = dialog[turn-1][0]
        
        src_candidate = [
            f" \n#instruction: any questions about this picture?\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
        ]
        weights = [1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 2
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def visdial_answer(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        caption = features['caption']
        dialog = json.loads(features['dialog'])
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"agent: let's talk about the picture."]
        for t in range(turn-1):
            context += [f"human: {dialog[t][0]}?"]
            context += [f"agent: {dialog[t][1]}."]
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]
        src_candidate = [
            f" \n#instruction: {question}\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {answer}",
        ]
        weights = [1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
    @staticmethod
    def visdial_caption(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        caption = features['caption']
        src_candidate = [
            f" \n#instruction: what does the image show?",
            f" \n#instruction: can you describe the image?",
        ]
        tgt_candidate = [
            f" {caption}",
            f" {caption}",
        ]
        weights = [0.5, 0.5]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== invig ======

    @staticmethod
    def invig_question(features, style=None):
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
        context = [f"human: {dialog[0][1]}"]
        for t in range(1, turn-1):
            context += [f"agent: {dialog[t][0]}"]  # question
            context += [f"human: {dialog[t][1]}"]  # answer
        question = dialog[turn-1][0]
        answer = f"{context[-1]} " if len(context) and random.random() > 0.5 else ""
        context = " ".join(context)
        
        src_candidate = [
            f" \n#instruction: ask a new question to guess what i want in this picture.\n#context: \"{context}\"",
            f" \n#instruction: {answer}can you specify which region the context describes?\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" not yet. {question}"
        ]
        weights = [9, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 2
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
        dialog = json.loads(features['dialog'])
        turn_right = len(dialog) - 1 if "?" not in dialog[-1][0] else len(dialog)  # 消除ok, i see.
        turn = random.randint(2, max(2, turn_right))  # 选一个轮数，来作为tgt_text
        context = [f"agent: {dialog[0][1]}"]
        for t in range(1, turn-1):
            context += [f"human: {dialog[t][0]}"]  # question
            context += [f"agent: {dialog[t][1]}"]  # answer
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]

        src_candidate = [
            f" \n#instruction: answer the question based on the region. {question}\n#region: {sbbox}\n#context: \"{context}\"",
            f" \n#instruction: {question}\n#region: {sbbox}\n#context: \"{context}\"",
            f" \n#instruction: what do you think of the region in the picture?\n#region: {sbbox}",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
            f" {dialog[0][1]}",
        ]
        weights = [5, 2, 3]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
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
        dialog = json.loads(features['dialog'])
        context = [f"human: {dialog[0][1]}"]
        for t in range(1, len(dialog)):
            context += [f"agent: {dialog[t][0]}"]  # question
            context += [f"human: {dialog[t][1]}"]  # answer
        context = " ".join(context)
        src_candidate = [
            f" \n#instruction: which region does the context describe?\n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes?\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" sure. region: {sbbox}",
        ]
        weights = [9, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and sbbox
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
## ====== guesswhat ======
    
    @staticmethod
    def guesswhat_question(features, style=None):
        """ question_invig """
        # 处理图片
        image = features['image']
        # 处理bbox
        # bbox = features.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # 处理文本
        dialog = json.loads(features['dialog'])
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"human: guess what i want."]
        for t in range(turn-1):
            context += [f"agent: {dialog[t][0]}"]  # question
            context += [f"human: {dialog[t][1]}"]  # answer
        question = dialog[turn-1][0]
        answer = f"{context[-1]} " if len(context) and random.random() > 0.5 else ""
        context = " ".join(context)
        
        src_candidate = [
            f" \n#instruction: {answer}ask a closed-ended question to guess what i want in this picture.\n#context: \"{context}\"",
            f" \n#instruction: {answer}can you specify which region the context describes?\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {question}",
            f" not yet. {question}"
        ]
        weights = [9, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and len(dialog) >= 1, f"{image}, {question}, {len(dialog)}"
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
        dialog = json.loads(features['dialog'])
        turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
        context = [f"agent: guess what i want."]
        for t in range(turn-1):
            context += [f"human: {dialog[t][0]}"]  # question
            context += [f"agent: {dialog[t][1]}"]  # answer
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]
        based_on_the_region = "based on the region " if random.random() > 0.5 else ""

        src_candidate = [
            f" \n#instruction: answer the question {based_on_the_region}with yes or no. {question}\n#region: {sbbox}\n#context: \"{context}\"",
            f" \n#instruction: {question} yes or no?\n#region: {sbbox}\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
        ]
        weights = [7, 3]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
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
        dialog = json.loads(features['dialog'])
        context = [f"human: guess what i want."]
        for t in range(0, len(dialog)):
            context += [f"agent: {dialog[t][0]}"]  # question
            context += [f"human: {dialog[t][1]}"]  # answer
        context = " ".join(context)
        src_candidate = [
            f" \n#instruction: which region does the context describe?\n#context: \"{context}\"",
            f" \n#instruction: can you specify which region the context describes?\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" region: {sbbox}",
            f" sure. region: {sbbox}",
        ]
        weights = [9, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and sbbox
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== ofa tasks ======

    # @staticmethod
    # def invig_grounding_ofa(features, weights=None):
    #     """ grounding_invig """
    #     # 处理图片
    #     image = features['image']
    #     # 处理bbox
    #     bbox = features['bbox']
    #     sbbox = bbox_to_sbbox(bbox, *image.size)
    #     # 处理文本
    #     dialog = json.loads(features['dialog'])
    #     context = [f"query: {dialog[0][1]}"]
    #     for t in range(1, len(dialog)):
    #         context += [f"question: {dialog[t][0]}"]
    #         context += [f"answer: {dialog[t][1]}"]
    #     context = " ".join(context)
    #     src_candidate = [
    #         f" which region does the text \" {context} \" describe?",
    #     ]
    #     tgt_candidate = [
    #         f" {sbbox}",
    #     ]
    #     style = 0
    #     src_text = src_candidate[style].lower()
    #     tgt_text = tgt_candidate[style].lower()
    #     assert image and sbbox
    #     return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    # @staticmethod
    # def guesswhat_answer_oracle(features, weights=None):
    #     """ answer_invig """
    #     # 处理图片
    #     image = features['image']
    #     # 处理bbox
    #     bbox = features['bbox']
    #     sbbox = bbox_to_sbbox(bbox, *image.size)
    #     # 处理文本
    #     dialog = json.loads(features['dialog'])
    #     turn = random.randint(1, len(dialog))  # 选一个轮数，来作为tgt_text
    #     context = f"query: guess what i want."
    #     question = dialog[turn-1][0]
    #     answer = dialog[turn-1][1]

    #     src_candidate = [
    #         f" \n#instruction: answer the question with yes or no. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\"",
    #     ]
    #     tgt_candidate = [
    #         f" {answer}",
    #     ]
    #     style = 0
    #     src_text = src_candidate[style].lower()
    #     tgt_text = tgt_candidate[style].lower()
    #     assert image and question and answer
    #     return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== objects365 ======

    @staticmethod
    def objects365_detection(features, style=None):
        """ 任务数据集：Grounding """
        # openimages
        known_cates = set(['Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ (Musical Instrument)', 'Cassette deck', 'Apple', 'Human eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Human beard', 'Bird', 'Parking meter', 'Traffic light', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll', 'Skull', 'Washing machine', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle', 'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones', 'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy', 'Screwdriver', 'Bicycle wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill (Tool)', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Brown bear', 'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot', 'Person', 'Bow and arrow', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat (Animal)', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Vehicle registration plate', 'Lantern', 'Toaster', 'Flashlight', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors', 'Stairs', 'Computer keyboard', 'Printer', 'Traffic sign', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock', 'Fire hydrant', 'Land vehicle', 'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Medical equipment', 'Cattle', 'Cello', 'Jet ski', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat', 'Bronze sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Computer mouse', 'Cookie', 'Office building', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Computer monitor', 'Box', 'Stapler', 'Christmas tree', 'Cowboy hat', 'Hiking equipment', 'Studio couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Human mouth', 'Dairy Product', 'Dice', 'Oven', 'Dinosaur', 'Ratchet (Device)', 'Couch', 'Cricket ball', 'Winter melon', 'Spatula', 'Whiteboard', 'Pencil sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Tin can', 'Mug', 'Tap', 'Harbor seal', 'Stretcher', 'Can opener', 'Goggles', 'Human body', 'Roller skates', 'Coffee cup', 'Cutting board', 'Blender', 'Plumbing fixture', 'Stop sign', 'Office supplies', 'Volleyball (Ball)', 'Vase', 'Slow cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Paper towel', 'Personal care', 'Food', 'Sun hat', 'Tree house', 'Flying disc', 'Skirt', 'Gas stove', 'Salt and pepper shakers', 'Mechanical fan', 'Face powder', 'Fax', 'Fruit', 'French fries', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'French horn', 'Window blind', 'Human foot', 'Golf cart', 'Jacket', 'Egg (Food)', 'Street light', 'Guitar', 'Pillow', 'Human leg', 'Isopod', 'Grape', 'Human ear', 'Power plugs and sockets', 'Panda', 'Giraffe', 'Woman', 'Door handle', 'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Baseball bat', 'Baseball glove', 'Mixing bowl', 'Marine invertebrates', 'Kitchen utensil', 'Light switch', 'House', 'Horse', 'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed', 'Adhesive tape', 'Harp', 'Sandal', 'Bicycle helmet', 'Saucer', 'Harpsichord', 'Human hair', 'Heater', 'Harmonica', 'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Drinking straw', 'Insect', 'Hair dryer', 'Kitchenware', 'Indoor rower', 'Invertebrate', 'Food processor', 'Bookcase', 'Refrigerator', 'Wood-burning stove', 'Punching bag', 'Common fig', 'Cocktail shaker', 'Jaguar (Animal)', 'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet', 'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Bottle opener', 'Lynx', 'Lavender (Plant)', 'Lighthouse', 'Dumbbell', 'Human head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 'Billiard table', 'Mammal', 'Mouse', 'Motorcycle', 'Musical instrument', 'Swim cap', 'Frying pan', 'Snowplow', 'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Waffle iron', 'Milk', 'Ring binder', 'Plate', 'Mobile phone', 'Baked goods', 'Mushroom', 'Crutch', 'Pitcher (Container)', 'Mirror', 'Personal flotation device', 'Table tennis racket', 'Pencil case', 'Musical keyboard', 'Scoreboard', 'Briefcase', 'Kitchen knife', 'Nail (Construction)', 'Tennis ball', 'Plastic bag', 'Oboe', 'Chest of drawers', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Hair spray', 'Sports equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Infant bed', 'Polar bear', 'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Digital clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'High heels', 'Red panda', 'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Submarine sandwich', 'Snowboard', 'Sword', 'Picture frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine', 'Scorpion', 'Segway', 'Training bench', 'Snake', 'Coffee table', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Picnic basket', 'Cooking spray', 'Trousers', 'Bowling equipment', 'Football helmet', 'Truck', 'Measuring cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Paper cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale', 'Zebra', 'Auto part', 'Jug', 'Pizza cutter', 'Cream', 'Monkey', 'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Toilet paper', 'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Wine glass', 'Countertop', 'Tablet computer', 'Waste container', 'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard', 'Axe', 'Hand dryer', 'Soap dispenser', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm tree', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Garden Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 'Horizontal bar', 'Convenience store', 'Bomb', 'Bench', 'Ice cream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange', 'Antelope', 'Beaker', 'Moths and butterflies', 'Window', 'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash (Plant)', 'Racket', 'Human face', 'Human arm', 'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Common sunflower', 'Microwave oven', 'Honeycomb', 'Marine mammal', 'Sea lion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Grinder', 'Spice rack', 'Light bulb', 'Corded phone', 'Sports uniform', 'Tennis racket', 'Wall clock', 'Serving tray', 'Kitchen & dining room table', 'Dog bed', 'Cake stand', 'Cat furniture', 'Bathroom accessory', 'Facial tissue holder', 'Pressure cooker', 'Kitchen appliance', 'Tire', 'Ruler', 'Luggage and bags', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band-aid', 'Animal', 'Bell pepper', 'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Human nose', 'Pen', 'Ant', 'Car', 'Aircraft', 'Human hand', 'Skunk', 'Teddy bear', 'Watermelon', 'Cantaloupe', 'Dishwasher', 'Flute', 'Balance beam', 'Sandwich', 'Shrimp', 'Sewing machine', 'Binoculars', 'Rays and skates', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe', 'Remote control', 'Wheelchair', 'Rugby ball', 'Armadillo', 'Maracas', 'Helmet'])
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
        weights = [2, 2, 3, 2, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== openimages_v1.2 ======

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
        weights = [2, 2, 3, 2, 1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}
    
## ====== cc_sbu_align ======

    @staticmethod
    def cc_sbu_align_caption(features, style=None):
        """ 任务数据集：Grounding """
        image = features['image']
        caption = features.get('caption')
        # 处理文本
        src_candidate = [
            f" \n#instruction: what does the image describe?",
        ]
        tgt_candidate = [
            f" {caption}",
        ]
        weights = [1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== llava_instruct_150k ======

    @staticmethod
    def llava_instruct_150k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['conversations']
        turn = random.randint(1, len(dialog)//2)
        context = []
        for i in range(turn-1):
            assert dialog[i*2]['from'] == 'human'
            context += ['human: ' + dialog[i*2]['value'].replace('<image>\n', '')]
            context += ['agent: ' + dialog[i*2+1]['value']]
        context = " ".join(context)
        question = dialog[(turn-1)*2]['value'].replace('<image>\n', '')
        answer = dialog[(turn-1)*2+1]['value']

        src_candidate = [
            f" \n#instruction: {question}\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {answer}",
        ]
        weights = [1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== llava_conversation_58k ======

    @staticmethod
    def llava_conversation_58k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['conversations']
        turn = random.randint(1, len(dialog)//2)
        context = []
        for i in range(turn-1):
            assert dialog[i*2]['from'] == 'human'
            context += ['human: ' + dialog[i*2]['value']]
            context += ['agent: ' + dialog[i*2+1]['value']]
        context = " ".join(context).replace('<image>\n', '')
        question = dialog[(turn-1)*2]['value'].replace('<image>\n', '')
        answer = dialog[(turn-1)*2+1]['value'].replace('<image>\n', '')

        src_candidate = [
            f" \n#instruction: {question}\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {answer}",
        ]
        weights = [1]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== llava_complex_reasoning_77k ======

    @staticmethod
    def llava_complex_reasoning_77k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['conversations']
        question = dialog[0]['value'].replace('<image>\n', '')
        answer = dialog[1]['value'].replace('<image>\n', '')
        step_by_step = " let's think step by step." if len(answer) > 120 else ""

        src_candidate = [
            f" \n#instruction: {question}",
            f" \n#instruction: {question}{step_by_step}",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
        ]
        weights = [0.5, 0.5]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

## ====== llava_detail_23k ======

    @staticmethod
    def llava_detail_23k(features, style=None):
        """ answer_invig """
        # 处理图片
        image = features['image']
        # 处理文本
        dialog = features['conversations']
        question = dialog[0]['value'].replace('<image>\n', '')
        answer = dialog[1]['value'].replace('<image>\n', '')
        in_detail = " Please provide a detailed description." if len(answer) > 100 else ""

        src_candidate = [
            f" \n#instruction: {question}",
            f" \n#instruction: {question}{in_detail}",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
        ]
        weights = [0.5, 0.5]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
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

