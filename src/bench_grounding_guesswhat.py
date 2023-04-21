# 基本设置
import argparse
import yaml
args = argparse.Namespace(**{
    "path_ofa": "/mnt/bn/hri-lq/projects/VLDD/OFA",
    "path_invig": "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig",
    "path_tokenizer": "/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large"
})

# 路径设置
import copy
import os
import sys
import re
import json
sys.path.append(args.path_ofa)
os.chdir(args.path_invig)

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
# from dataclasses import dataclass
# from torch.cuda.amp import autocast
import logging
import numpy as np
import torch
import random

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{args.path_invig}/src"}))

# 载入transformers
from src.ofa.tokenization_ofa import OFATokenizer
# from transformers.modeling_outputs import Seq2SeqLMOutput
# from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

        
# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from src.dialog_dataset import TaskProcessor, DataCollatorForOFA, OFADataset
from src.dialog_dataset import LoadImage
from src.common import world_info_from_env, get_image_processor, sbbox_to_bbox
from src.common import sbbox_to_bbox
from src.evaluation import eval_grounding_acc_v2
from src.common import OFAModelWarper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

with open('/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_4ds.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    main_args = parser.parse_args()

    # 1 load model
    path_ckpt = "/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_4_21000.pt"
    path_ckpt = main_args.ckpt
    model = OFAModelWarper(path_ckpt, args.path_ofa, args.path_tokenizer)
    logger.info(f"Test Oracle Acc on {path_ckpt}")

    # 2 load dataset
    path_dataset = "/mnt/bn/hri-lq/datasets/hf/guesswhat-success"
    split = "test"
    ds = TaskProcessor.setup_task(load_from_disk(path_dataset)[split], "guesswhat_grounding").shuffle(42)
    ds_ofa = OFADataset(ds, config['path_images'], model.tokenizer, model.image_processor)
    loader = torch.utils.data.DataLoader(ds_ofa, batch_size=main_args.batch_size, num_workers=4, collate_fn=ds_ofa.collater)

    # 3 benchmark
    t = tqdm(loader)
    sbbox_gt, sbbox_gen = [], []
    bbox_gt, bbox_gen = [], []
    for i, b in enumerate(t):
        sbbox_gt = model.tokenizer.batch_decode(b['target'])
        sbbox_gen = model.generate_origin(b)
        bbox_gt += [sbbox_to_bbox(s) for s in sbbox_gt]
        bbox_gen += [sbbox_to_bbox(s) for s in sbbox_gen]
        if i % 5 == 0:
            res = eval_grounding_acc_v2(np.stack(bbox_gt), np.stack(bbox_gen))
            t.set_postfix({"ap50": res['grounding_acc_iou_0.5']})
        # break
    bbox_gt = np.stack(bbox_gt)
    bbox_gen = np.stack(bbox_gen)
    acc = eval_grounding_acc_v2(bbox_gt, bbox_gen)
    logger.info(f"GndingAcc: {acc}")

    # 4 save results
    with open("./eval_grounding_guesswhat.log", 'w') as f:
        json.dump({"acc": acc, "bbox_gt": bbox_gt.tolist(), "bbox_gen": bbox_gen.tolist()}, f)
    logger.info(f"Saved results to ./eval_grounding_guesswhat.log")
