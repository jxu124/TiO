# 基本设置
import argparse
import yaml

path_yaml = "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_env.yml"
with open(path_yaml) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 路径设置
import copy
import os
import sys
import re
import json
sys.path.append(cfg['env']['path_ofa'])
os.chdir(cfg['env']['path_invig'])

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
# from dataclasses import dataclass
# from torch.cuda.amp import autocast
import logging
import numpy as np
import torch
import random
from PIL import Image
import mmcv

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{cfg['env']['path_invig']}/ofa_invig"}))

# 载入transformers
from src.transformers_ofa.tokenization_ofa import OFATokenizer
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from src.legacy.dialog_dataset import MapFunc, DataCollatorForOFA, get_dataset
from src.common import world_info_from_env, get_processor, sbbox_to_bbox, bbox_to_sbbox
from src.common import OFAModelWarper
from src.evaluation import eval_grounding_acc_v2


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


class MyCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        bboxes_gt = []
        bboxes_gen = []
        for b in kwargs['eval_dataloader']:
            b['net_input'].pop('prev_output_tokens')
            sbboxes_gt = tokenizer.batch_decode(b.pop('target'))
            sbboxes_gen = model.generate_origin(b, dtype=model.dtype)
            bboxes_gt += [sbbox_to_bbox(s) for s in sbboxes_gt]
            bboxes_gen += [sbbox_to_bbox(s) for s in sbboxes_gen]
        res = eval_grounding_acc_v2(np.asarray(bboxes_gen), np.asarray(bboxes_gt))
        trainer.log_metrics("eval", res)
        kwargs['metrics']['eval_ap50'] = res['grounding_acc_iou_0.5']
        return kwargs['metrics']

if __name__ == "__main__":
    DEBUG_MODE = True
    # DEBUG_MODE = False

    MapFunc.load_config(path_yaml)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--bf16", default=False, action="store_true")
    parser.add_argument("--fsdp", default=False, action="store_true")
    parser.add_argument("--deepspeed", default=None, type=str)
    # args = parser.parse_args(["-c", "/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt"])
    # args = parser.parse_args(["-c", "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230427-1312/checkpoint_1_2000.pt"])
    args = parser.parse_args()

    # 1 load model
    path_ckpt = args.checkpoint
    model = OFAModelWarper(path_ckpt, cfg['env']['path_ofa'], path_yaml)
    tokenizer, image_processor = MapFunc.processor
    logger.info(f"Train from {path_ckpt}")

    # 2 load dataset
    ds_train = get_dataset('train', distributed_split=False)
    ds_valid = get_dataset('validation', distributed_split=False)
    # ds_valid.ds = ds_valid.ds.select(range(128))

    # 3 setting
    hyper_args = argparse.Namespace(**{
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-5,
        "num_train_epochs": 10,
        "warmup_steps": 3000,  # 默认为线性 lr_scheduler
    })
    setting_args = argparse.Namespace(**{
        "fp16": not args.bf16,
        "fp16_full_eval": not args.bf16,
        "bf16": args.bf16,
        "bf16_full_eval": args.bf16,
        "deepspeed": args.deepspeed,
        "dataloader_num_workers": 4,
        "logging_steps": 10 if DEBUG_MODE else 50,
        "report_to": "none" if DEBUG_MODE else "all",
        "output_dir": "./results-debug" if DEBUG_MODE else "./results/hf_huge/",
        "label_names": ["target"],
        "evaluation_strategy": "steps",
        "eval_steps": 100 if DEBUG_MODE else 1000,
        "save_strategy": "steps",
        "save_steps": 500 if DEBUG_MODE else 1000,
        "save_total_limit": 20,
        "remove_unused_columns": False,
        "fsdp": ["full_shard", "auto_wrap"] if not args.deepspeed else False,
        "ddp_find_unused_parameters": True if not args.deepspeed else False,
        # "load_best_model_at_end": True,
    })

    trainer = transformers.Trainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=DataCollatorForOFA(tokenizer=tokenizer, image_processor=image_processor),
        args=transformers.TrainingArguments(**vars(hyper_args), **vars(setting_args)),
        # callbacks=[MyCallback()]
    )
    trainer.train()


