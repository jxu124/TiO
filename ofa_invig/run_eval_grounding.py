# 基本设置
import argparse
args = argparse.Namespace(**{
    "path_ofa": "/mnt/bn/hri-lq/projects/VLDD/OFA",
    "path_invig": "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig",
})

# 路径设置
import os
import sys
sys.path.append(args.path_ofa)
os.chdir(args.path_invig)

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
from dataclasses import dataclass
from torch.cuda.amp import autocast
import logging
import numpy as np
import torch
import random

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{args.path_invig}/src"}))

# 载入transformers
from src.ofa.tokenization_ofa import OFATokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from src.dialog_dataset import TaskProcessor
from src.common import world_info_from_env, get_image_processor, sbbox_to_bbox
from src.evaluation import eval_grounding_acc_v2


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")



def load_from_pretrain(ckpt_path):
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


def _ofa_generator(model, sample, dtype=torch.float16, device='cuda'):
    model.eval()
    model.to(device=device)
    for k, v in sample['net_input'].items():
        sample['net_input'][k] = v.to(device=device, dtype=dtype) if v.dtype == torch.float32 else v.to(device=device)
    with autocast(dtype=dtype):
        with torch.no_grad():
            hypos = task.inference_step(generator, [model], sample)
    hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
    return tokenizer.batch_decode(hypos)


def ofa_generator(model, src_texts: List[str], images: List[Any], dtype=torch.float16, device='cuda'):
    src_inputs = tokenizer(src_texts, return_length=True, padding=True, return_tensors="pt")
    src_tokens = src_inputs.input_ids
    src_lengths = src_inputs.length
    patch_images = torch.stack([image_processor(i) for i in images])
    patch_masks = torch.tensor([True] * len(src_tokens))
    sample = {"net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths, 'patch_images': patch_images, 'patch_masks': patch_masks}}
    return _ofa_generator(model, sample, dtype=torch.float16, device='cuda')


# model warper - be a transformers-like module
class OFAModelWarper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, **batch):
        net_output = self.model(**batch['net_input'])
        lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
        target = self.model.get_targets(batch, net_output)
        loss = torch.nn.functional.cross_entropy(lprobs.view(-1, 59457), target.view(-1), 
                                                 ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=0.1)
        return Seq2SeqLMOutput(loss=loss, logits=lprobs)
        

# @dataclass
# class DataCollatorForOFA(transformers.DataCollatorForSeq2Seq):
#     tokenizer: transformers.PreTrainedTokenizerBase
#     image_processor: Optional[Any] = None
#     padding: Optional[str] = "longest"
#     max_src_length: Optional[int] = 256
#     max_tgt_length: Optional[int] = 64
#     return_tensors: Optional[str] = "pt"
    
#     def __call__(self, features, return_tensors=None):
#         src_texts = [f[0] for f in features]
#         tgt_texts = [f[1] for f in features]
#         patch_images = torch.stack([self.image_processor(f[2]) for f in features])
        
#         # max_length longest
#         inputs = self.tokenizer(text=src_texts, return_length=True, max_length=self.max_src_length, 
#                                 padding=self.padding, truncation=True, return_tensors=self.return_tensors)
#         labels = self.tokenizer(text_target=tgt_texts, max_length=self.max_tgt_length, 
#                                 padding=self.padding, truncation=True, return_tensors=self.return_tensors)
#         patch_masks = torch.tensor([True] * len(patch_images))
#         prev_output_tokens = labels.input_ids[..., :-1]
#         target = labels.input_ids[..., 1:]
#         features = {
#             "net_input": {
#                 'src_tokens': inputs.input_ids,
#                 "src_lengths": inputs.length,
#                 'patch_images': patch_images,
#                 'patch_masks': patch_masks,
#                 "prev_output_tokens": prev_output_tokens
#             },
#             "target": target
#         }
#         return features
    

# class OFADataset(torch.utils.data.Dataset):
#     def __init__(self, dataset):
#         # 分布式适配
#         local_rank, rank, world_size = world_info_from_env()
#         self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        
#     def __getitem__(self, index):
#         if type(index) is not int:
#             raise NotImplementedError
#         else:
#             data = self.dataset.__getitem__(index)
#             data = TaskProcessor.auto_task(data)
#         return data
    
#     def __len__(self):
#         return len(self.dataset)
    