# Author: Jie Xu
# Date: 4/2/2023

from io import BytesIO

import logging
import warnings
import math

import numpy as np
import torch
import base64
import utils.transforms as T

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset


# 参数设置
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

PATH_TO_OPENIMAGES = "/mnt/bn/hri-lq/datasets/openimages_v1.2"
PATH_TO_COCO = "/mnt/bn/hri-lq/datasets/coco"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# from transformers import OFATokenizer
# path_to_tokenckpt = "/mnt/bn/hri-lq/projects/OFA/test/eval-invig-model-large"
# tokenizer = OFATokenizer.from_pretrained(path_to_tokenckpt, local_files_only=True, truncation_side="left")


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

    prefix_tokens = None
    if samples[0].get("decoder_prompt", None) is not None:
        prefix_tokens = merge("decoder_prompt")
        prefix_tokens = prefix_tokens[:, 1:]

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

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

    return batch


class OFADialogDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=300,
        max_tgt_length=60,
        patch_image_size=480,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.dataset = dataset
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        self.prob_grounding = 0.5
        self.prob_question = 0.5
        self.prob_grounding = 0.33

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = ' which region does the text " {} " describe?'
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = '这段文字" {} "描述的是哪个区域？'

    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[-max_ques_words:])

        return question
    
    def __getitem__(self, index):
        def get_qa_text(his, qas, turn=math.inf, ttype="g"):
            assert len(qas) > 0
            assert ttype in ["g", "q", "a"]  # ttype in ["g", "q", "a"]
            history = his
            i = -1
            for i in range(min(len(qas)-1, turn)):
                history = f"{history} q: {qas[i]['q']} a: {qas[i]['a']}"
            i += 1
            # print("len(qas), i:", len(qas), i)
            if ttype == "q":
                text_src = history
                text_tgt = qas[i]['q'].strip()
            elif ttype == "a":
                text_src = f"{history} q: {qas[i]['q']}"
                text_tgt = qas[i]['a'].strip()
            elif ttype == "g":
                text_src = f"{history} q: {qas[i]['q']} q: {qas[i]['a']}"
                text_tgt = ""
            text_src, text_tgt = text_src.strip().lower(), text_tgt.strip().lower()
            return text_src, text_tgt

        # 1 读取数据项目
        # item = {"id": dialogue_id, "img": path_to_img, "prompt": history, "qas": qas, "bbox": [], "cate": ""}
        item = self.dataset[index]
        uniq_id, image_path, history, qas, region_coord, obj_cate = item['id'], item['img'], item['prompt'], item['qas'], item['bbox'], item['cate']

        # 1.1 选择任务

        # 2 history, qas 转化为 text_src, text_tgt
        prob_grounding = self.prob_grounding
        has_bbox = (len(region_coord) == 4)
        if np.random.rand(1)[0] < prob_grounding and has_bbox:
            turn = math.inf
            ttype = "g"
            text_src, text_tgt = get_qa_text(history, qas, turn=turn, ttype=ttype)
        else:
            turn = np.random.randint(len(qas))
            ttype = "q" if np.random.rand(1)[0] < self.prob_question else "a"  # qa的选择五五开
            text_src, text_tgt = get_qa_text(history, qas, turn=turn, ttype=ttype)

        # 3 载入图片
        PATH_IMGS_PAIR = [("invig", PATH_TO_OPENIMAGES), ("guesswhat", PATH_TO_COCO), ("visdial", PATH_TO_COCO)]
        for a, b in PATH_IMGS_PAIR:
            if uniq_id.startswith(a):
                path_imgs_dir = b
                break
        image = Image.open(f"{path_imgs_dir}/{image_path}").convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        
        # grounding
        if not has_bbox:
            region_coord = [0, 0, w-1, h-1]
        x0, y0, x1, y1 = region_coord
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)

        
        # 问题截断
        text_src = self.pre_question(text_src, self.max_src_length)

        # 回答 prev_output_item是用于教师强制的参数对应每个解码器的输入
        if ttype == "q":
            # src
            src_item = self.encode_text(text_src)
            src_prompt = self.encode_text("Output a question. ")
            src_item = torch.cat([self.bos_item, src_prompt, src_item, self.eos_item])
            # tgt
            tgt_item = self.encode_text(f"q: {text_tgt}")
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = prev_output_item[0:0]
        elif ttype == "a":
            # src
            if has_bbox:
                src_prompt = self.encode_text(f"Output an answer. Given {region_coord} . ")
            else:
                src_prompt = self.encode_text("Output an answer. ")
            src_item = self.encode_text(text_src)
            src_item = torch.cat([self.bos_item, src_prompt, src_item, self.eos_item])
            # tgt
            tgt_item = self.encode_text(f"a: {text_tgt}")
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = prev_output_item[0:0]
        elif ttype == "g":
            # src
            src_item = self.encode_text(text_src)
            src_prompt = self.encode_text("Output the region described. ")
            src_item = torch.cat([self.bos_item, src_prompt, src_item, self.eos_item])
            # tgt
            tgt_item = torch.cat([
                self.encode_text(f"g:"),
                self.encode_text(f"{region_coord}", use_bpe=False)
            ])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = prev_output_item[0:0]

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region,
            "decoder_prompt": decoder_prompt
        }

        return example

    # def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
    #     目前存在halffloat的bug, 暂缓修复
    #     length = 240  # left chunk
    #     return tokenizer.encode(text, add_special_tokens=False, max_length=length, truncation=True, return_tensors="pt").squeeze(0)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)


## ==== utils ====
import json
import re
import os
import torch
import warnings
import jsonlines
import numpy as np


PATH_TO_OPENIMAGES = "/mnt/bn/hri-lq/datasets/openimages_v1.2"
PATH_TO_COCO = "/mnt/bn/hri-lq/datasets/coco"
PATH_TO_INVIGTS = "/mnt/bn/hri-lq/datasets/invig/invig_v0.1_anns/anns/dialog_translation.json"
PATH_TO_INVIG = "/mnt/bn/hri-lq/datasets/invig/invig_v0.1_anns/anns"
PATH_TO_VISDIAL = "/mnt/bn/hri-lq/datasets/VisDial"
PATH_TO_GUESSWHAT = "/mnt/bn/hri-lq/datasets/guesswhat"
SEP_Q = "<sep_q>"
SEP_A = "<sep_a>"


xyxy_to_xywh = lambda bbox: bbox - np.concatenate([np.zeros_like(bbox[..., :2]), bbox[..., :2]], axis=-1)
xywh_to_xyxy = lambda bbox: bbox + np.concatenate([np.zeros_like(bbox[..., :2]), bbox[..., :2]], axis=-1)


def parse_dialog_en(dialog):
    dialog = re.split(f'{SEP_Q}|{SEP_A}', dialog)[1:]
    dialog = [s.strip() for s in dialog]
    ref_exp = dialog[0]
    q = []
    a = []
    for i, s in enumerate(dialog[1:]):
        if i % 2 == 0:
            q.append(s)
        else:
            a.append(s)
    qas = zip(q, a)
    qas = [{'q': q.strip(), 'a': a.strip()} for q, a in qas]
    return ref_exp, qas


# sbbox to bbox
def sbbox_to_bbox(sbbox, w, h, num_bins=1000, strict=False):
    res = re.match(".*?<bin_([0-9]*)> <bin_([0-9]*)> <bin_([0-9]*)> <bin_([0-9]*)>.*", sbbox)
    if res is None:
        if strict:
            return None
        else:
            return np.array([0, 0, w-1, h-1])
    bbox = np.asarray([int(res.group(1))/num_bins, int(res.group(2))/num_bins, int(res.group(3))/num_bins, int(res.group(4))/num_bins])
    bbox = np.asarray([int(res.group(1))/(num_bins-1), int(res.group(2))/(num_bins-1), int(res.group(3))/(num_bins-1), int(res.group(4))/(num_bins-1)])
    bbox = bbox * np.array([w, h, w, h])
    return bbox

    
def bbox_to_sbbox(bbox, w, h, num_bins=1000):
    bbox = np.asarray(bbox) / np.asarray([w, h, w, h])
    quant_x0 = "<bin_{}>".format(int((bbox[0] * (num_bins - 1)).round()))
    quant_y0 = "<bin_{}>".format(int((bbox[1] * (num_bins - 1)).round()))
    quant_x1 = "<bin_{}>".format(int((bbox[2] * (num_bins - 1)).round()))
    quant_y1 = "<bin_{}>".format(int((bbox[3] * (num_bins - 1)).round()))
    region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
    return region_coord
