
# builtin
import re
import os
import random
import logging
from typing import Optional, Any, Union, List

# 3rd-part
from PIL import Image, ImageFile
import torch
import numpy as np
import torchvision.transforms as T

from .ofa.tokenization_ofa import OFATokenizer

# ========== logger ==========

logger = logging.getLogger(__name__)


# ========== 预定义的一些参数 ==========
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


# # model warper
# class OFAModelWarper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
        
#     def forward(self, **batch):
#         net_output = self.model(**batch['net_input'])
#         lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
#         target = self.model.get_targets(batch, net_output)
#         loss = torch.nn.functional.cross_entropy(lprobs.view(-1, 59457), target.view(-1), 
#                                                  ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=0.1)
#         return Seq2SeqLMOutput(loss=loss, logits=lprobs)
    

# ========== 预处理函数 ==========
# 获得 tokenizer 和 image_processor
def get_image_processor(resolution=512, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    image_processor = T.Compose([
        lambda image: image.convert("RGB"),
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return image_processor


def get_processor(pretrain_path="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large", 
                  resolution=512, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    tokenizer = OFATokenizer.from_pretrained(pretrain_path, local_files_only=True, truncation_side="right")
    image_processor = get_image_processor(resolution, mean, std)
    return tokenizer, image_processor


# ========== 工具函数 ==========

# sbbox 转换 bbox
def sbbox_to_bbox(sbbox, w=None, h=None, num_bins=1000, strict=False):
    """ This function converts a string bounding box (sbbox) to a dense bounding box (bbox). """
    res = re.match(r".*?<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>.*", sbbox)
    if res is None:
        if strict:
            import warnings
            warnings.warn(f"Cannot parse bbox from \"{sbbox}\" .")
        bbox = np.asarray([0, 0, 999, 999])
    else:
        bbox = np.asarray([int(res.group(1))/num_bins, int(res.group(2))/num_bins, int(res.group(3))/num_bins, int(res.group(4))/num_bins])
    if w is not None and h is not None: 
        bbox = bbox * np.array([w, h, w, h])
    return bbox

    
def bbox_to_sbbox(bbox, w, h, num_bins=1000):
    """ This function converts a dense bounding box (bbox) to a string bounding box (sbbox). """
    bbox = np.asarray(bbox).reshape(4) / np.asarray([w, h, w, h])
    bbox = np.clip(bbox, 1e-3, 1 - 1e-3)
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


# def _ofa_generator(gens, sample, dtype=torch.float16, device='cuda'):
#     from torch.cuda.amp import autocast
#     # tokenizer image_processor
#     generator, model, cfg, task = gens
#     model.eval()
#     model.to(device=device)
#     for k, v in sample['net_input'].items():
#         sample['net_input'][k] = v.to(device=device, dtype=dtype) if v.dtype == torch.float32 else v.to(device=device)
#     with autocast(dtype=dtype):
#         with torch.no_grad():
#             hypos = task.inference_step(generator, [model], sample)
#     hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
#     return tokenizer.batch_decode(hypos)


# def ofa_generator(gens, src_texts: List[str], images: List[Any], dtype=torch.float16, device='cuda'):
#     src_inputs = tokenizer(src_texts, return_length=True, padding=True, return_tensors="pt")
#     src_tokens = src_inputs.input_ids
#     src_lengths = src_inputs.length
#     patch_images = torch.stack([image_processor(i) for i in images])
#     patch_masks = torch.tensor([True] * len(src_tokens))
#     sample = {"net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths, 'patch_images': patch_images, 'patch_masks': patch_masks}}
#     return _ofa_generator(gens, sample, dtype=torch.float16, device='cuda')


def load_from_pretrain(ckpt_path):
    try:
        from utils import checkpoint_utils
    except:
        raise ImportError("Cannot import 'utils', please set OFA path to sys.path.")
    logger.info("开始载入model参数...")
    model_overrides = {"bpe_dir": f"{args.path_ofa}/utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
    model = models[0]
    logger.info("开始载入ema参数...")
    # model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
    logger.info("模型设置中...")
    model.eval()
    # model.prepare_for_inference_(cfg)
    generator = task.build_generator(models, cfg.generation)
    logger.info("完成！")
    return generator, model, cfg, task


class OFAModelWarper(torch.nn.Module):
    def __init__(self, ckpt_path, ofa_path, tokenizer_path, resolution=512):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.ofa_path = ofa_path
        self.tokenizer_path = tokenizer_path
        self.gens = self.load_from_pretrain(ckpt_path, ofa_path)
        _, self.model, _, task = self.gens
        self.tokenizer, self.image_processor = task.processor
         # = OFATokenizer.from_pretrained(tokenizer_path, local_files_only=True, truncation_side="right")
         # = get_image_processor(resolution)
        
    def forward(self, **batch):
        from transformers.modeling_outputs import Seq2SeqLMOutput
        net_output = self.model(**batch['net_input'])
        lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
        target = self.model.get_targets(batch, net_output)
        loss = torch.nn.functional.cross_entropy(lprobs.view(-1, 59457), target.view(-1), 
                                                 ignore_index=self.tokenizer.pad_token_id, reduction='mean', label_smoothing=0.1)
        return Seq2SeqLMOutput(loss=loss, logits=lprobs)

    @staticmethod
    def load_from_pretrain(ckpt_path, ofa_path):
        try:
            from utils import checkpoint_utils
        except:
            raise ImportError("Cannot import 'utils', please set OFA path to sys.path.")
        logger.info("开始载入model参数...")
        model_overrides = {"bpe_dir": f"{ofa_path}/utils/BPE"}
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
        model = models[0]
        logger.info("开始载入ema参数...")
        # model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        logger.info("模型设置中...")
        model.eval()
        # model.prepare_for_inference_(cfg)
        generator = task.build_generator(models, cfg.generation)
        logger.info("完成！")
        return generator, model, cfg, task
    
    def generate_origin(self, sample, dtype=torch.float16, device='cuda'):
        from torch.cuda.amp import autocast
        # tokenizer image_processor
        generator, model, cfg, task = self.gens
        model.eval()
        model.to(device=device)
        for k, v in sample['net_input'].items():
            sample['net_input'][k] = v.to(device=device, dtype=dtype) if v.dtype == torch.float32 else v.to(device=device)
        with autocast(dtype=dtype):
            with torch.no_grad():
                hypos = task.inference_step(generator, [model], sample)
        hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
        return self.tokenizer.batch_decode(hypos)

    def generate(self, src_texts: List[str], images: List[Any], dtype=torch.float16, device='cuda'):
        src_inputs = self.tokenizer(src_texts, return_length=True, padding=True, return_tensors="pt")
        src_tokens = src_inputs.input_ids
        src_lengths = src_inputs.length
        patch_images = torch.stack([self.image_processor(i) for i in images])
        patch_masks = torch.tensor([True] * len(src_tokens))
        sample = {"net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths, 'patch_images': patch_images, 'patch_masks': patch_masks}}
        return self.generate_origin(sample, dtype=torch.float16, device='cuda')

    