import numpy as np
import re
import os
import torch


# sbbox to bbox
def sbbox_to_bbox(sbbox, w, h, num_bins=1000):
    res = re.match("[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>(.*)", sbbox)
    if res is None:
        return None
    bbox = np.asarray([int(res.group(1))/num_bins, int(res.group(2))/num_bins, int(res.group(3))/num_bins, int(res.group(4))/num_bins])
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


from transformers import OFATokenizer
import random
import torchvision.transforms as T
from PIL import Image


patch_image_size = 480
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
image_processor = T.Compose([
    T.Resize([patch_image_size, patch_image_size]),
    # T.Resize([patch_image_size], max_size=patch_image_size),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

PATH_TO_TF = "/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large"
tokenizer = OFATokenizer.from_pretrained(PATH_TO_TF, local_files_only=True, truncation_side="right")


def get_texts_from_dialog(dialog, sbbox=None):
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
    canditates = [("q", 0), ("q", 1), ("a", 0), ("g", 0), ("g", 1)]
    weights = [0.222, 0.109, 0.33, 0.222, 0.109]
    if sbbox is not None:
        task = random.choices(canditates, weights=weights, k=1)[0]
    else:
        task = random.choices(canditates[:-2], weights=weights[:-2], k=1)[0]
    task_map = {"q": get_texts_from_dialog_q, "a": get_texts_from_dialog_a, "g": get_texts_from_dialog_g}
    task_fn = task_map[task[0]]
    src_text, tgt_text = task_fn(dialog=dialog, sbbox=sbbox, style=task[1])
    return src_text, tgt_text
    

def get_texts_from_dialog_q(dialog, style=0, **args):
    st = 1 if dialog[0][0] != "" else min(2, len(dialog))
    turn = random.randint(st, len(dialog))
    context = []
    for t in range(turn-1):
        if dialog[t][0] != "":
            context += [f"question: {dialog[t][0]}"]
        context += [f"answer: {dialog[t][1]}"]
    context = " ".join(context)
    question = dialog[turn-1][0]
    
    if style == 0:
        src_text=f" \n#instruction: ask a question to find the object based on the context. \n#context:\"{context}\""
        tgt_text=f" {question}"
    else:
        src_text=f" \n#instruction: can you give the region described by the context? \n#context:\"{context}\""
        tgt_text=f" not yet. {question}"
    return src_text, tgt_text


def get_texts_from_dialog_a(dialog, sbbox=None, style=0):
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
    
    if style == 0:
        if sbbox is None:
            src_text=f" \n#instruction: {question} answering based on the context. \n#context:\"{context}\""
        else:
            src_text=f" \n#instruction: {question} answering based on the region and context. \n#region:{sbbox}\n#context:\"{context}\""
        tgt_text=f" {answer}"
    else:
        raise NotImplemented
    return src_text, tgt_text


def get_texts_from_dialog_g(dialog, sbbox, style=0):
    assert sbbox is not None
    turn = len(dialog)
    context = []
    for t in range(turn):
        if dialog[t][0] != "":
            context += [f"question: {dialog[t][0]}"]
        context += [f"answer: {dialog[t][1]}"]
    context = " ".join(context)
    
    if style == 0:
        src_text=f" \n#instruction: which region does the context describe? \n#context:\"{context}\""
        tgt_text=f" region:" + sbbox
    else:
        src_text=f" \n#instruction: can you give the region described by the context? \n#context:\"{context}\""
        tgt_text=f" yes. region:" + sbbox
    return src_text, tgt_text


def collate_ofa(batch):
    
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
    
    # print(src_texts, tgt_texts)
    # src_inputs = tokenizer(src_texts, text_target=tgt_texts, add_special_tokens=True, return_length=True, max_length=120, padding="longest", return_tensors="pt")
    # src_inputs = tokenizer.prepare_seq2seq_batch(
    #     src_texts, tgt_texts, return_length=True, max_target_length=120, padding="longest", return_tensors="pt")
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
    # batch = {
    #     "id": id,
    #     "nsentences": len(samples),
    #     "ntokens": ntokens,
    #     "net_input": {
    #         "src_tokens": src_tokens,
    #         "src_lengths": src_lengths,
    #         "patch_images": patch_images,
    #         "patch_masks": patch_masks,
    #         "prev_output_tokens": prev_output_tokens
    #     },
    #     "decoder_prompts": decoder_prompts,
    #     "target": target,
    #     "prefix_tokens": prefix_tokens,
    #     "w_resize_ratios": w_resize_ratios,
    #     "h_resize_ratios": h_resize_ratios,
    #     "region_coords": region_coords
    # }
    return batch
