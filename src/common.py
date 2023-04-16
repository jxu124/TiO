
# builtin
import re
import os
import random

# 3rd-part
from PIL import Image, ImageFile
import torch
import numpy as np
import torchvision.transforms as T

from .ofa.tokenization_ofa import OFATokenizer


# ========== 预定义的一些参数 ==========
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


# ========== 预处理函数 ==========
# 获得 tokenizer 和 image_processor
def get_image_processor(resolution=480, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    image_processor = T.Compose([
        lambda image: image.convert("RGB"),
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return image_processor


def get_processor(pretrain_path="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large", 
                  resolution=480, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    tokenizer = OFATokenizer.from_pretrained(pretrain_path, local_files_only=True, truncation_side="right")
    image_processor = T.Compose([
        lambda image: image.convert("RGB"),
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return tokenizer, image_processor


# ========== 工具函数 ==========

# sbbox 转换 bbox
def sbbox_to_bbox(sbbox, w=None, h=None, num_bins=1000):
    """ This function converts a string bounding box (sbbox) to a dense bounding box (bbox). """
    res = re.match(".*?<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>.*", sbbox)
    if res is None:
        bbox = np.asarray([0, 0, 999, 999])
    else:
        bbox = np.asarray([int(res.group(1))/num_bins, int(res.group(2))/num_bins, int(res.group(3))/num_bins, int(res.group(4))/num_bins])
    if w is not None and h is not None: 
        bbox = bbox * np.array([w, h, w, h])
    return bbox

    
def bbox_to_sbbox(bbox, w, h, num_bins=1000):
    """ This function converts a dense bounding box (bbox) to a string bounding box (sbbox). """
    bbox = np.asarray(bbox).reshape(4) / np.asarray([w, h, w, h])
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
