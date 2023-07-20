# ***
# ***

# builtin library
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Any, Union, List
import warnings
import random

# 3rd library & self library
import numpy as np
from .data import bbox_to_sbbox


class DataPrompt:
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
        """
        # max_turn(human_first) - [3, 5, 7, 9, ...] -> [0, 1, 2, 3]
        # max_turn(agent_first) - [2, 4, 6, 8, ...] -> [0, 1, 2, 3]
        max_turn = (len(dialog) - 1) // 2 - 1 if human_first else len(dialog) // 2 - 1
        if min_turn is None:
            turn = max_turn
        else:
            if min_turn > max_turn:
                warnings.warn(f"min_turn: {min_turn}, max_turn: {max_turn}")
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
        src_text = DataPrompt.make_src_text(instruction, context, src_sbbox, human_first).replace("  ", " ")
        tgt_text = (" " + tgt_text.strip()).replace("  ", " ")
        return src_text, tgt_text

# ====== refcoco ======

    @staticmethod
    def refcoco_caption(features, style=None):
        """  Grounding """
        # ***
        image = features['image']
        # ***
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        caption = random.choice(features['captions'])
        dialogs = [["describe the region briefly.", caption]]
        text_candidate = [DataPrompt.make_text_pair(d, src_sbbox=sbbox) for d in dialogs]

        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def refcoco_grounding(features, style=None):
        """  Grounding """
        # ***
        image = features['image']
        # ***
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        caption = random.choice(features['captions'])
        dialogs = [["which region does the context describe?", caption, f"region: {sbbox}"]]
        text_candidate = [DataPrompt.make_text_pair(d, human_first=True) for d in dialogs]

        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== visdial_question ======

    @staticmethod
    def visdial_question(features, style=None):
        """ question_invig """
        # ***
        image = features['image']
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0] + '?', i[1] + '.']]
        dialogs = [['ask a question based on the image.', *dialog]]
        text_candidate = [DataPrompt.make_text_pair(d, 0) for d in dialogs]

        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def visdial_answer(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0] + '?', i[1] + '.']]
        dialogs = [['answer the question.', *dialog]]
        text_candidate = [DataPrompt.make_text_pair(d, 0, human_first=True) for d in dialogs]

        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def visdial_caption(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        caption = features['caption']
        dialogs = [['what does the image show?', caption],
                   ['describe the image briefly.', caption]]
        text_candidate = [DataPrompt.make_text_pair(d) for d in dialogs]

        weights = [1, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== invig ======

    @staticmethod
    def invig_question(features, style=None):
        """ question_invig """
        # ***
        image = features.get('image', None)
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i][1:]
        if "?" not in dialog[-2] and len(dialog) > 3:  # ***
            dialog = dialog[:-2]
        dialogs = [["ask a question to guess what i want.", *dialog],
                   ["can you specify which region the context describes?", *dialog]]
        text_candidate = [DataPrompt.make_text_pair(d, 0, human_first=True) for d in dialogs]

        weights = [9, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " not yet." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def invig_answer(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i][1:]
        if "?" not in dialog[-2] and len(dialog) > 3:  # ***
            dialog = dialog[:-2]
        dialogs = [["answer the question based on the region.", *dialog],
                   ["state your query about the region.", dialog[0]]]
        text_candidate = [DataPrompt.make_text_pair(dialogs[0], 1, src_sbbox=sbbox), DataPrompt.make_text_pair(dialogs[1], 0, src_sbbox=sbbox)]

        weights = [9, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        # tgt_text = "not yet. " + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def invig_grounding(features, style=None):
        """ grounding_invig """
        # ***
        image = features['image']
        # ***
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0], i[1]]][1:]
        if "?" not in dialog[-2] and len(dialog) > 2:  # ***
            dialog = dialog[:-2]
        dialogs = [["which region does the context describe?", *dialog, f" region: {sbbox}"],
                   ["can you specify which region the context describes?", *dialog, f" region: {sbbox}"]]
        text_candidate = [DataPrompt.make_text_pair(d, human_first=True) for d in dialogs]

        weights = [9, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " sure." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== guesswhat ======

    @staticmethod
    def guesswhat_question(features, style=None):
        """ question_invig """
        # ***
        image = features['image']
        # ***
        # bbox = features.get('bbox', None)
        # sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialogs = [["ask a close-ended question to guess what i want."] + dialog,
                   ["can you specify which region the context describes?"] + dialog]
        text_candidate = [DataPrompt.make_text_pair(d, 0) for d in dialogs]

        weights = [9, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " not yet." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def guesswhat_answer(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialogs = [["answer the question based on the region with yes or no."] + dialog]
        text_candidate = [DataPrompt.make_text_pair(d, 0, human_first=True, src_sbbox=sbbox) for d in dialogs]

        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        # tgt_text = "not yet. " + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

    @staticmethod
    def guesswhat_grounding(features, style=None):
        """ grounding_invig """
        # ***
        image = features['image']
        # ***
        bbox = features['bbox']
        sbbox = bbox_to_sbbox(bbox, *image.size)
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in [i[0], i[1]]]
        dialogs = [["which region does the context describe?"] + dialog + [f" region: {sbbox}"],
                   ["can you specify which region the context describes?"] + dialog + [f" region: {sbbox}"]]
        text_candidate = [DataPrompt.make_text_pair(d) for d in dialogs]

        weights = [9, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        tgt_text = " sure." + tgt_text if style == 1 else tgt_text
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== objects365 ======

    @staticmethod
    def objects365_detection(features, style=None):
        raise NotImplementedError

# ====== openimages_v1.2 ======

    @staticmethod
    def openimages_detection(features, style=None):
        """  Grounding """
        # ***
        known_cates = set(['Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ (Musical Instrument)', 'Cassette deck', 'Apple', 'Human eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Human beard', 'Bird', 'Parking meter', 'Traffic light', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll', 'Skull', 'Washing machine', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle', 'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones', 'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy', 'Screwdriver', 'Bicycle wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill (Tool)', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Brown bear', 'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot', 'Person', 'Bow and arrow', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat (Animal)', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Vehicle registration plate', 'Lantern', 'Toaster', 'Flashlight', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors', 'Stairs', 'Computer keyboard', 'Printer', 'Traffic sign', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock', 'Fire hydrant', 'Land vehicle', 'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Medical equipment', 'Cattle', 'Cello', 'Jet ski', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat', 'Bronze sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Computer mouse', 'Cookie', 'Office building', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Computer monitor', 'Box', 'Stapler', 'Christmas tree', 'Cowboy hat', 'Hiking equipment', 'Studio couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Human mouth', 'Dairy Product', 'Dice', 'Oven', 'Dinosaur', 'Ratchet (Device)', 'Couch', 'Cricket ball', 'Winter melon', 'Spatula', 'Whiteboard', 'Pencil sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Tin can', 'Mug', 'Tap', 'Harbor seal', 'Stretcher', 'Can opener', 'Goggles', 'Human body', 'Roller skates', 'Coffee cup', 'Cutting board', 'Blender', 'Plumbing fixture', 'Stop sign', 'Office supplies', 'Volleyball (Ball)', 'Vase', 'Slow cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Paper towel', 'Personal care', 'Food', 'Sun hat', 'Tree house', 'Flying disc', 'Skirt', 'Gas stove', 'Salt and pepper shakers', 'Mechanical fan', 'Face powder', 'Fax', 'Fruit', 'French fries', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'French horn', 'Window blind', 'Human foot', 'Golf cart', 'Jacket', 'Egg (Food)', 'Street light', 'Guitar', 'Pillow', 'Human leg', 'Isopod', 'Grape', 'Human ear', 'Power plugs and sockets', 'Panda', 'Giraffe', 'Woman', 'Door handle', 'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Baseball bat', 'Baseball glove', 'Mixing bowl', 'Marine invertebrates', 'Kitchen utensil', 'Light switch', 'House', 'Horse', 'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed', 'Adhesive tape', 'Harp', 'Sandal', 'Bicycle helmet', 'Saucer', 'Harpsichord', 'Human hair', 'Heater', 'Harmonica', 'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Drinking straw', 'Insect', 'Hair dryer', 'Kitchenware', 'Indoor rower', 'Invertebrate', 'Food processor', 'Bookcase', 'Refrigerator', 'Wood-burning stove', 'Punching bag', 'Common fig', 'Cocktail shaker', 'Jaguar (Animal)', 'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet', 'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Bottle opener', 'Lynx', 'Lavender (Plant)', 'Lighthouse', 'Dumbbell', 'Human head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 'Billiard table', 'Mammal', 'Mouse', 'Motorcycle', 'Musical instrument', 'Swim cap', 'Frying pan', 'Snowplow', 'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Waffle iron', 'Milk', 'Ring binder', 'Plate', 'Mobile phone', 'Baked goods', 'Mushroom', 'Crutch', 'Pitcher (Container)', 'Mirror', 'Personal flotation device', 'Table tennis racket', 'Pencil case', 'Musical keyboard', 'Scoreboard', 'Briefcase', 'Kitchen knife', 'Nail (Construction)', 'Tennis ball', 'Plastic bag', 'Oboe', 'Chest of drawers', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Hair spray', 'Sports equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Infant bed', 'Polar bear', 'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Digital clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'High heels', 'Red panda', 'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Submarine sandwich', 'Snowboard', 'Sword', 'Picture frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine', 'Scorpion', 'Segway', 'Training bench', 'Snake', 'Coffee table', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Picnic basket', 'Cooking spray', 'Trousers', 'Bowling equipment', 'Football helmet', 'Truck', 'Measuring cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Paper cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale', 'Zebra', 'Auto part', 'Jug', 'Pizza cutter', 'Cream', 'Monkey', 'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Toilet paper', 'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Wine glass', 'Countertop', 'Tablet computer', 'Waste container', 'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard', 'Axe', 'Hand dryer', 'Soap dispenser', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm tree', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Garden Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 'Horizontal bar', 'Convenience store', 'Bomb', 'Bench', 'Ice cream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange', 'Antelope', 'Beaker', 'Moths and butterflies', 'Window', 'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash (Plant)', 'Racket', 'Human face', 'Human arm', 'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Common sunflower', 'Microwave oven', 'Honeycomb', 'Marine mammal', 'Sea lion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Grinder', 'Spice rack', 'Light bulb', 'Corded phone', 'Sports uniform', 'Tennis racket', 'Wall clock', 'Serving tray', 'Kitchen & dining room table', 'Dog bed', 'Cake stand', 'Cat furniture', 'Bathroom accessory', 'Facial tissue holder', 'Pressure cooker', 'Kitchen appliance', 'Tire', 'Ruler', 'Luggage and bags', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band-aid', 'Animal', 'Bell pepper', 'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Human nose', 'Pen', 'Ant', 'Car', 'Aircraft', 'Human hand', 'Skunk', 'Teddy bear', 'Watermelon', 'Cantaloupe', 'Dishwasher', 'Flute', 'Balance beam', 'Sandwich', 'Shrimp', 'Sewing machine', 'Binoculars', 'Rays and skates', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe', 'Remote control', 'Wheelchair', 'Rugby ball', 'Armadillo', 'Maracas', 'Helmet'])
        # ***
        image = features['image']
        # ***
        anns_info = features['anns_info']
        bboxes = np.asarray([i["bbox"] for i in anns_info])
        categorys = [i["category"] for i in anns_info]
        c, num = random.choice(Counter(categorys).most_common())
        mask = [i == c for i in categorys]
        bboxes = bboxes[np.where(mask)].tolist()
        # bboxes.sort(key=lambda a: a[0] * 100 + a[1])
        # sbboxes = [bbox_to_sbbox(bbox) for bbox in bboxes]
        sbboxes = list(set([bbox_to_sbbox(b) for b in sorted(np.asarray(bboxes).tolist())]))
        # ***
        num_less = num - random.randint(0, num - 1)
        num_more = num + random.randint(1, 4)
        # ***
        false_c = random.choice(list(known_cates - set(categorys)))
        c = c.replace("/", " or ")
        false_c = false_c.replace("/", " or ")

        dialogs = [[f"point out the location of any {num_less} {c}.", f" there are {num} {c}. here are {num_less} of them.\n" + "\n".join([f"- region: {sbbox}" for sbbox in random.choices(sbboxes, k=num_less)])],
                   [f"point out the location of any {num_more} {c}.", f" there are only {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes])],
                   [f"where are {c} located?", f" there are {num} {c}.\n" + "\n".join([f" region: {sbbox}" for sbbox in sbboxes])],
                   [f"how many {c} are there? where are they located?", f" there are {num} {c}.\n" + "\n".join([f"- region: {sbbox}" for sbbox in sbboxes])],
                   [f"where are {false_c} located?", f" there is no {false_c}."],
                   [f"which region does the context describe? #context: {c}", "\n".join([f" region: {sbbox}" for sbbox in sbboxes])]]
        text_candidate = [DataPrompt.make_text_pair(d) for d in dialogs]

        weights = [2, 2, 3, 2, 1]
        weights = [0, 0, 1, 0, 1, 9]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== cc_sbu_align ======

    @staticmethod
    def cc_sbu_align_caption(features, style=None):
        """  Grounding """
        image = features['image']
        caption = features.get('caption')
        # ***
        dialogs = [['what does the image show?', caption],
                   ["describe the image.", caption]]
        text_candidate = [DataPrompt.make_text_pair(d) for d in dialogs]
        weights = [1, 1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_instruct_150k ======

    @staticmethod
    def llava_instruct_150k(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialog = ["answer the question.", *dialog]
        text_candidate = [DataPrompt.make_text_pair(dialog, 0, human_first=True)]
        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_conversation_58k ======

    @staticmethod
    def llava_conversation_58k(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        dialog = ["answer the question.", *dialog]
        text_candidate = [DataPrompt.make_text_pair(dialog, 0, human_first=True)]
        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_complex_reasoning_77k ======

    @staticmethod
    def llava_complex_reasoning_77k(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        text_candidate = [DataPrompt.make_text_pair(dialog)]
        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== llava_detail_23k ======

    @staticmethod
    def llava_detail_23k(features, style=None):
        """ answer_invig """
        # ***
        image = features['image']
        # ***
        dialog = features['dialog']
        dialog = [j for i in dialog for j in i]
        text_candidate = [DataPrompt.make_text_pair(dialog)]
        weights = [1]
        style = style if style is not None else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        return {"src_text": src_text, "tgt_text": tgt_text, "image": image}

# ====== end of TaskProcess ======
