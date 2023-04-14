
# builtin
import re
import os
import random

# 3rd-part
from PIL import Image, ImageFile
import torch
import numpy as np
import torchvision.transforms as T

# OFA
from transformers import OFATokenizer


# ========== 预定义的一些参数 ==========
resolution = 480
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


# ========== 预处理函数 ==========
# 获得 tokenizer 和 image_processor
def get_processor(pretrain_path="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large"):
    tokenizer = OFATokenizer.from_pretrained(pretrain_path, local_files_only=True, truncation_side="right")
    image_processor = T.Compose([
        lambda image: image.convert("RGB"),
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return tokenizer, image_processor


# 我需要确定的 
# prompt_q: context question(GT)
prompt_q = [
    {
        "src_text": " \n#instruction: ask a context-specific question to identify the object.\n#context:\"{context}\"",
        "tgt_text": " {question}"
    },
    {
        "src_text": " \n#instruction: tell me which region does the context describe?\n#context:\"{context}\"",
        "tgt_text": " I'm not sure. {question}"
    }
]
# prompt_a: context question sbbox(optional) answer(GT)
prompt_a = [
    {
        "src_text": " \n#instruction: answer these. {question}\n#context:\"{context}\"",
        "tgt_text": " {answer}"
    },
    {
        "src_text": " \n#instruction: answer these. {question}\n#region:{sbbox}\n#context:\"{context}\"",
        "tgt_text": " {answer}"
    }
]
# prompt_g: context sbbox(GT)
prompt_g = [
    {
        "src_text": " \n#instruction: which region does the context describe?\n#context:\"{context}\"",
        "tgt_text": " region:{sbbox}"
    },
    {
        "src_text": " \n#instruction: tell me which region does the context describe?\n#context:\"{context}\"",
        "tgt_text": " sure. region:{sbbox}"
    }
]


# ========== 定义一些 collate_fn 函数 ==========
def collate_ofa(batch):
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
    src_texts, tgt_texts, patch_images = [], [], []
    for data in batch:
        image = data.get('image', None)
        if image is None:
            image = Image.open(data['image_path'])
        image = image.convert("RGB")
        w, h = image.size
        bbox = data.get('bbox', None)
        sbbox = bbox_to_sbbox(bbox, w, h) if bbox is not None else None
        src_text, tgt_text = get_texts_from_dialog(data['dialog'], sbbox)
        
        src_texts += [src_text]
        tgt_texts += [tgt_text]
        patch_images += [image_processor(image)]
    patch_images = torch.stack(patch_images)
    
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


def get_texts_from_dialog(dialog, sbbox=None, weights={"q": 0.33, "a": 0.33, "g": 0.34}):
    """
    Example:
    Q.0: 0.33 * 0.66
    src_text=" ask a question to find the object based on the context. \n context: {...}"
    tgt_text=" {question}"
    Q.1: 0.33 * 0.33
    src_text=" can you give the region described by the context? \n context: {...}"
    tgt_text=" not yet. {question}"
    A.0: 0.33
    src_text=" {question} based on the context and region. \n context: {...} \n region:" + region_coord_item
    tgt_text=" {answer}"
    G.0: 0.33 * 0.66
    src_text=" which region does the context describe? \n context: {...}"
    tgt_text=" region:" + region_coord_item
    G.1: 0.33 * 0.33
    src_text=" can you give the region described by the context? \n context: {...}"
    tgt_text=" yes. region:" + region_coord_item
    """
    if sbbox is None:
        weights['g'] = 0
    canditates = list(weights.keys())
    weights = np.asarray([weights[k] for k in canditates])
    weights = weights / np.sum(weights)  # 和为1
    task = random.choices(canditates, weights=weights, k=1)[0]

    task_map = {"q": get_texts_from_dialog_q, "a": get_texts_from_dialog_a, "g": get_texts_from_dialog_g}
    task_fn = task_map[task]
    src_text, tgt_text = task_fn(dialog=dialog, sbbox=sbbox)
    return src_text, tgt_text


def get_texts_from_dialog_q(dialog, **args):
    st = 1 if dialog[0][0] != "" else min(2, len(dialog))
    turn = random.randint(st, len(dialog))
    context = []
    for t in range(turn-1):
        if dialog[t][0] != "":
            context += [f"question: {dialog[t][0]}"]
        context += [f"answer: {dialog[t][1]}"]
    context = " ".join(context)
    question = dialog[turn-1][0]
    
    style = random.randint(0, 1)
    src_text = prompt_q[style]['src_text'].format(context=context)
    tgt_text = prompt_q[style]['tgt_text'].format(question=question)
    return src_text, tgt_text


def get_texts_from_dialog_a(dialog, sbbox=None):
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
    
    style = 0 if sbbox is None else 1
    src_text = prompt_a[style]['src_text'].format(context=context, question=question, sbbox=sbbox)
    tgt_text = prompt_a[style]['tgt_text'].format(answer=answer)
    return src_text, tgt_text


def get_texts_from_dialog_g(dialog, sbbox):
    assert sbbox is not None
    turn = len(dialog)
    context = []
    for t in range(turn):
        if dialog[t][0] != "":
            context += [f"question: {dialog[t][0]}"]
        context += [f"answer: {dialog[t][1]}"]
    context = " ".join(context)
    
    style = random.randint(0, 1)
    src_text = prompt_g[style]['src_text'].format(context=context)
    tgt_text = prompt_g[style]['tgt_text'].format(sbbox=sbbox)
    return src_text, tgt_text


# ========== 工具函数 ==========

# sbbox 转换 bbox
def sbbox_to_bbox(sbbox, w=None, h=None, num_bins=1000):
    """ This function converts a string bounding box (sbbox) to a dense bounding box (bbox). """
    res = re.match("[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>(.*)", sbbox)
    if res is None:
        return None
    bbox = np.asarray([int(res.group(1))/num_bins, int(res.group(2))/num_bins, int(res.group(3))/num_bins, int(res.group(4))/num_bins])
    if w is not None and h is not None: 
        bbox = bbox * np.array([w, h, w, h])
    return bbox

    
def bbox_to_sbbox(bbox, w, h, num_bins=1000):
    """ This function converts a dense bounding box (bbox) to a string bounding box (sbbox). """
    bbox = np.asarray(bbox) / np.asarray([w, h, w, h])
    quant_x0 = "<bin_{}>".format(int((bbox[0] * (num_bins - 1)).round()))
    quant_y0 = "<bin_{}>".format(int((bbox[1] * (num_bins - 1)).round()))
    quant_x1 = "<bin_{}>".format(int((bbox[2] * (num_bins - 1)).round()))
    quant_y1 = "<bin_{}>".format(int((bbox[3] * (num_bins - 1)).round()))
    region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
    return region_coord


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size
